import copy
import datetime
import logging
import os
import csv
import time
from fvcore.common.timer import Timer
import numpy as np

import mtilt.utils.comm as comm
from mtilt.evaluation.testing import flatten_results_dict, print_csv_format
from mtilt.utils.events import EventStorage, EventWriter

from .train_loop import HookBase

__all__ = [
    "IterationTimer",
    "PeriodicWriter",
    "EvalHook",
]
logger = logging.getLogger(__name__)

"""
Implement some common hooks.
"""


class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary in the end of training.

    This hook uses the time between the call to its :meth:`before_step`
    and :meth:`after_step` methods.
    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    """

    def __init__(self, warmup_iter=3):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()
        self._start_time = time.perf_counter()
        self._total_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.storage.iter + 1 - self.trainer.start_epoch

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):

        sec = self._step_timer.seconds()
        self.trainer.storage.put_scalars(time=sec)

        self._total_timer.pause()


class PeriodicWriter(HookBase):
    """
    Write events to EventStorage (by calling ``writer.write()``) periodically.

    It is executed every ``period`` iterations and after the last iteration.
    Note that ``period`` does not affect how data is smoothed by each writer.
    """

    def __init__(self, writers, period=20):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w
        self._period = period

    def after_step(self):
        if (self.trainer.epoch + 1) % self._period == 0 or (
            self.trainer.epoch == self.trainer.max_epoch - 1
        ):
            for writer in self._writers:
                writer.write()

    def after_train(self):
        for writer in self._writers:
            # If any new data is found (e.g. produced by other after_train),
            # write them before closing
            writer.write()
            writer.close()


class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_function):
        """
        Args:
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """

        self._func = eval_function
        self._metrics = []

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                if isinstance(v, np.ndarray):
                    continue
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            results_backup = copy.deepcopy(flattened_results)
            if "pw" in flattened_results:
                flattened_results.pop("pw")
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)
            self._metrics.append(results_backup)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_step(self):

        self._do_eval()

    def after_train(self):

        metrics_dict = {
            k: np.mean([x[k] for x in self._metrics], axis=0) for k in self._metrics[0].keys()
        }

        logger.info("Final average metrics: \n ")
        print_csv_format(metrics_dict)

        pws = [metric["pw"].tolist() for metric in self._metrics]
        with open(f'{self.trainer.cfg.OUTPUT_DIR}/pw.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for i, pw in enumerate(pws):
                csv_writer.writerows(pw)
                if i < len(pws) - 1:
                    csv_writer.writerow([])

        del self._func
