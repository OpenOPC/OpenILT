import argparse
import sys
import logging
import os
from typing import Optional, OrderedDict

import torch
from omegaconf import OmegaConf

from ..utils import comm
from ..utils.file_io import PathManager
from ..utils.logger import setup_logger
from ..utils.collect_env import collect_env_info
from ..config import LazyConfig, CfgNode
from ..utils.env import seed_all_rng
from .train_loop import TrainerBase, SimpleTrainer
from ..optimizer import  build_scaler
from ..solving import (
    build_solver,
    build_litho,
    build_initializer,
    build_algorithm,
    build_objectives
)
from ..utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from . import hooks
from ..evaluation.testing import verify_results, print_csv_format
from ..evaluation import ICCAD2013Evaluator, DatasetEvaluator
from ..data import MetadataCatalog

def default_argument_parser(epilog=None):
    """
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--allow-unsafe", default=False, help="allow unsafe yaml syntax")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default

def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code

def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
            )
        )

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            logger.info("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn data has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )


def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    PathManager.mkdirs(output_dir)
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]

class DefaultTrainer(TrainerBase):
    def __init__(self, cfg):

        super().__init__()
        logger = logging.getLogger("mtilt")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        objectives = build_objectives(cfg)
        self.litho_model = build_litho(cfg)
        initializer = build_initializer(cfg)
        solver = build_solver(cfg)

        if cfg.ALGORITHM.MULTI_TASK_LEARNING.MTL:
            mtl_scaler = self.build_scaler(cfg)
        else:
            mtl_scaler = None
        algorithm = build_algorithm(cfg, mtl_scaler)


        data_loader = self.build_train_loader(cfg)
        max_iter = cfg.SOLVER.MAX_ITER
        self._trainer = SimpleTrainer(
            solver, data_loader, objectives, self.litho_model, initializer, algorithm, max_iter, self.epoch
        )

        self.start_epoch = 0
        self.max_epoch = len(data_loader)
        self.cfg = cfg

        self.register_hooks(self.build_hooks())


    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_epoch, self.max_epoch)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def run_step(self):
        self._trainer.epoch = self.epoch
        self._trainer.model.training = True
        self._trainer.run_epoch(self.cfg)


    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()

        ret = [
            hooks.IterationTimer(),
        ]

        def evaluate_and_save_results():
            self._last_eval_results = self.evaluate(cfg)
            return self._last_eval_results

        ret.append(hooks.EvalHook(evaluate_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=1))
        return ret


    @classmethod
    def build_evaluator(cls, cfg, benchmark_name, output_folder=None):
        """
        todo
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(benchmark_name).evaluator_type
        if evaluator_type == "iccad2013":
            evaluator_list.append(ICCAD2013Evaluator(benchmark_name, cfg, output_folder))
        else:
            raise KeyError("now we only support iccad2013 benchmark evaluation.")

        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluator(evaluator_list)


    def evaluate(self, cfg, evaluators=None):
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TRAIN) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TRAIN), len(evaluators)
            )

        results = OrderedDict()
        for idx, benchmark_name in enumerate(cfg.DATASETS.TRAIN):
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = DefaultTrainer.build_evaluator(cfg, benchmark_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[benchmark_name] = {}
                    continue

            self._trainer.model.training = False  # for evaluate mode
            results_i = evaluator.evaluate(self._trainer.save_dir, self.epoch, self._trainer.model)
            results[benchmark_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(benchmark_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_epoch)

    # @classmethod
    # def build_lr_scheduler(cls, cfg, optimizer):
    #     """
    #     It now calls :func:`detectron2.solver.build_lr_scheduler`.
    #     Overwrite it if you'd like a different scheduler.
    #     """
    #     return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_scaler(cls, cfg):
        """
        todo
        """
        return build_scaler(cfg)

    @classmethod
    def build_train_loader(cls, cfg):

        raise NotImplementedError

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        When the config is defined for certain number of workers (according to
        ``cfg.SOLVER.REFERENCE_WORLD_SIZE``) that's different from the number of
        workers currently in use, returns a new cfg where the total batch size
        is scaled so that the per-GPU batch size stays the same as the
        original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.

        Other config options are also scaled accordingly:
        * training steps and warmup steps are scaled inverse proportionally.
        * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.

        Returns:
            CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
        """
        old_world_size = cfg.TRAINER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
                cfg.TRAINER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.TRAINER.IMS_PER_BATCH = int(round(cfg.TRAINER.IMS_PER_BATCH * scale))
        lr = cfg.TRAINER.BASE_LR = cfg.TRAINER.BASE_LR * scale
        max_iter = cfg.TRAINER.MAX_ITER = int(round(cfg.TRAINER.MAX_ITER / scale))
        warmup_iter = cfg.TRAINER.WARMUP_ITERS = int(round(cfg.TRAINER.WARMUP_ITERS / scale))
        cfg.TRAINER.STEPS = tuple(int(round(s / scale)) for s in cfg.TRAINER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.TRAINER.CHECKPOINT_PERIOD = int(round(cfg.TRAINER.CHECKPOINT_PERIOD / scale))
        cfg.TRAINER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg