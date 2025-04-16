import logging
import os
import datetime
import torch
torch.multiprocessing.set_start_method('spawn')

from mtilt.engine import default_argument_parser, launch, default_setup
from mtilt.config import get_cfg
from mtilt.utils import comm
from mtilt.utils.logger import setup_logger
from mtilt.engine.defaults import DefaultTrainer
from mtilt.utils.events import EventStorage
from mtilt.evaluation.testing import verify_results
from mtilt.data import build_train_loader, BenchmarkMapper


class Trainer(DefaultTrainer):

    def train_loop(self, start_epoch: int, max_epoch: int):
        """
        Args:
            start_iter, max_iter (int): See introductions.md above
        """
        logger = logging.getLogger("mtilt.trainer")
        logger.info("Starting doing ILT...")

        self.epoch = self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        with EventStorage(start_epoch) as self.storage:
            self.before_train()
            for self.epoch in range(start_epoch, max_epoch):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_epoch, self.max_epoch)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        """

        return build_train_loader(cfg, mapper=BenchmarkMapper(cfg))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file, args.allow_unsafe)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg

def main(args):
    """
    the overall pipeline: only involved three steps.
    """
    # 1. build config
    cfg = setup(args)

    # 2. build solver
    trainer = Trainer(cfg)

    # 3. solve ILT
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
