import copy
import os
import shutil
import time
from typing import Mapping
import cv2
import logging

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from mtilt.config import CfgNode
from mtilt.utils.events import get_event_storage
import mtilt.utils.comm as comm
from .build import META_SOLVER_REGISTRY



logger = logging.getLogger(__name__)

@META_SOLVER_REGISTRY.register()
class AtomicSolver():

    lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
    bestParams = None
    bestMask = None

    def __init__(self, config: CfgNode):

        self.config = config
        # just for saving index
        if self.config.LITHO_OPERATOR.MASK:
            mask_index_savedir = os.path.join(self.config.OUTPUT_DIR, "mask_index")
            if os.path.exists(mask_index_savedir):
                shutil.rmtree(mask_index_savedir)
            os.makedirs(mask_index_savedir)

    def _reset_cls_vars(self):

        self.lossMin, self.pvbMin = 1e12, 1e12
        self.bestMask = None
        self.bestParams = None

    def _copy_loss_dict(self, loss_dict):

        new_loss_dict = {}
        for key, value in loss_dict.items():
            if isinstance(value, (torch.Tensor)):
                new_loss_dict[key] = value.detach().clone()
            elif isinstance(value, list):
                tmp_list = []
                for v in value:
                    tmp_list.append(v.detach().clone())
                new_loss_dict[key] = tmp_list
            else:
                new_loss_dict[key] = copy.deepcopy(value)

        return new_loss_dict

    def __call__(self, inputs, algo, objectives, model, iters, sample_num=0):

        self._reset_cls_vars()

        backup = inputs["backup"]
        params = inputs["params"]
        target = inputs["target"]

        # instantiate algorithm.optimizer
        algo.bind_optimizer(value=params)
        if algo.mtl:
            algo.bind_scaler(params=params)
        model.bind_backup(backup)
        loss_dict = {}
        l2_losses = []

        start_sample = time.perf_counter()
        for index in range(iters):

            start = time.perf_counter()

            #
            mask, printedNoms, printedMax, printedMin, pvband = model(params, num=sample_num)

            #
            loss_dict = objectives.losses(mask, printedNoms, printedMax, printedMin, target)
            loss_dict_for_show = self._copy_loss_dict(loss_dict)

            #
            losses = algo(loss_dict, params, iteration=index, num=sample_num)

            spend_time = time.perf_counter() - start

            if self.bestParams is None or \
                    self.bestMask is None or \
                    losses.item() < self.lossMin:
                self.lossMin, self.pvbMin = losses, pvband

                self.bestParams = params.detach().clone()
                self.bestMask = mask.detach().clone()

            logger.info(f"[Iteration: {index}]: Litho_loss = {self.lossMin.item():.0f}; PVBand: {pvband.item():.0f}")
            self._write_metrics(loss_dict_for_show, spend_time)

            l2_losses.append([loss_dict_for_show[k].item() for k in loss_dict_for_show.keys() if "l2_loss_" in k])

            savedir = os.path.join(self.config.OUTPUT_DIR, "results_per_iter", f"{sample_num}")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            torch.save(mask, f"{savedir}/iter_{index}_mask.pkl")
        torch.save(target, f"{savedir}/target.pkl")

        #
        # spend_time_sample = time.perf_counter() - start_sample
        # with open(os.path.join(self.config.OUTPUT_DIR, "spend_time.txt"), 'a') as file:
        #     file.write(str(spend_time_sample)+"\n")
        #
        # logger.info("Start saving loss plots on {}th case.".format(sample_num))
        # import pandas as pd
        # df = pd.DataFrame(np.array(l2_losses).T, columns=[f'col_{i + 1}' for i in range(np.array(l2_losses).T.shape[1])])
        #
        # df.to_csv('loss.csv', index=False, float_format='%.4f')
        #
        # if not self.config.LITHO_OPERATOR.MASK:
        #     self.save_dir = os.path.join(self.config.OUTPUT_DIR, "l2loss_plots")
        #     if not os.path.exists(self.save_dir):
        #         os.makedirs(self.save_dir)
        #
        #     plt.figure(figsize=(10, 6))
        #     plot_data = np.array(l2_losses).T
        #
        #     for i in range(len(plot_data)):
        #         plt.plot(range(1, plot_data.shape[1]+1), plot_data[i]/self.config.SOLVER.WEIGHT_L2, label=f'corer {i} l2_loss')
        #
        #     plt.legend()
        #     plt.title(f'corners l2_loss levelset') # {self.config.ALGORITHM.CORNERS_LIST[0]}
        #     plt.xlabel('Iteration')
        #     plt.ylabel('Value')
        #     plt.grid(True)
        #     plt.savefig(os.path.join(self.save_dir, f"{sample_num}.jpg"))
        #     plt.close()

        # exit()
        return loss_dict, self.bestMask, \
               self.bestParams, self.pvbMin


    def _write_metrics(self,
                       loss_dict: Mapping[str, torch.Tensor],
                       spend_time: float,
                       prefix: str = "",
                       ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            spend_time (float): time taken by optimization iteration
            prefix (str): prefix for logging keys
        """
        for index, value in enumerate(loss_dict["l2_loss"]):
            loss_dict[f"l2_loss_corner{index}"] = value * len(loss_dict["l2_loss"])
        loss_dict.pop("l2_loss")
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["spend_time"] = spend_time

        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            spend_time = np.max([x.pop("spend_time") for x in all_metrics_dict])
            storage.put_scalar("spend_time", spend_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }

            total_losses_reduced = metrics_dict['litho_loss']
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_ilt_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

