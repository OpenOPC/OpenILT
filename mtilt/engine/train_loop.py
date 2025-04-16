# -*- coding: utf-8 -*-
import concurrent.futures
import logging
import numpy as np
import time
import os
import weakref
from typing import List, Mapping, Optional
import cv2
import torch

import mtilt.utils.comm as comm
from mtilt.utils.events import EventStorage, get_event_storage
from mtilt.utils.logger import _log_api_usage

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    """

    trainer: "TrainerBase" = None
    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_backward(self):
        """
        Called after the backward pass of each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self) -> None:
        self._hooks: List[HookBase] = []
        self.epoch: int = 0
        self.start_epoch: int = 0
        self.max_epoch: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_epoch: int, max_epoch: int):
        """
        Args:
            start_iter, max_iter (int): See introductions.md above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_epoch))

        self.epoch = self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        with EventStorage(start_epoch) as self.storage:
            try:
                self.before_train()
                for self.epoch in range(start_epoch, max_epoch):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.epoch += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.epoch = self.epoch
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        self.storage.epoch = self.epoch

        for h in self._hooks:
            h.before_step()

    def after_backward(self):
        for h in self._hooks:
            h.after_backward()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(
        self,
        solver,
        data_loader,
        objectives,
        litho_model,
        initializer,
        algorithm,
        max_iter,
        gather_metric_period=1,
        zero_grad_before_forward=True,
        async_write_metrics=False,
    ):
        """
        todo
        """
        super().__init__()


        self.solver = solver
        self.data_loader = data_loader
        # to access the data loader iterator, call `self._data_loader_iter`
        self._data_loader_iter_obj = None
        self.objectives = objectives
        self.model = litho_model
        self.initializer = initializer
        self.algorithm = algorithm
        self.max_iter = max_iter

        self.gather_metric_period = gather_metric_period
        self.zero_grad_before_forward = zero_grad_before_forward
        self.async_write_metrics = async_write_metrics
        # create a thread pool that can execute non critical logic in run_step asynchronically
        # use only 1 worker so tasks will be executred in order of submitting.
        self.concurrent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def run_epoch(self, cfg):
        """
        Implement the standard training logic described above.
        """

        # 1. load design
        inputs = next(self._data_loader_iter)

        # 2. do optimization
        loss_dict, bestMask, bestParams, pvbMin = \
            self.solver(inputs[0], self.algorithm, self.objectives, self.model, self.max_iter, self.epoch)

        # 3. save results
        self._save_results_epoch(cfg.OUTPUT_DIR, bestMask, inputs[0]["ref_target"])


        self.after_backward()


    def _save_results_epoch(self, dir, mask, target):

        logger = logging.getLogger(__name__)
        logger.info("Start saving results on {}th case.".format(self.epoch))

        self.save_dir = os.path.join(dir, "results")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        cv2.imwrite(f"{self.save_dir}/MOSAIC_test{self.epoch}.png", (mask * 255).detach().cpu().numpy())
        torch.save(mask, f"{self.save_dir}/{self.epoch}_bestmask.pkl")
        torch.save(target, f"{self.save_dir}/{self.epoch}_target.pkl")


    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj


    def after_train(self):
        super().after_train()
        self.concurrent_executor.shutdown(wait=True)
