import functools
import glob
import inspect
import os
import math

import torch
import torch.nn as nn

from mtilt.config import configurable, CfgNode
from mtilt.utils.events import get_event_storage


def get_objectives_num(backward_func=None):

    if backward_func is not None:
        assert (
            inspect.isfunction(backward_func)
            and backward_func.__name__ == "backward"
        ), "Incorrect use of @get_objectives_num"

        @functools.wraps(backward_func)
        def wrapped(self, *args, **kwargs):

            losses = kwargs.get('losses', {})
            if losses:
                self.task_num = len(losses)

            losses = backward_func(self, *args, **kwargs)
            return losses

        return wrapped

class GradScalerBase(nn.Module):

    """ An abstract class for gradient scaling strategies.
    todo
    """


    def __init__(
            self,
            config: CfgNode,
            params: torch.Tensor
        ):
        super().__init__()

        self.cfg = config
        self._params = params
        self.device = config.LITHO_OPERATOR.DEVICE

        self.mode = self.cfg.ALGORITHM.MULTI_TASK_LEARNING.MODE

        self.use_mask = self.cfg.ALGORITHM.MULTI_TASK_LEARNING.MASK.USE_MASK
        self.mask_mode = self.cfg.ALGORITHM.MULTI_TASK_LEARNING.MASK.MODE
        self.mask_ratio = self.cfg.ALGORITHM.MULTI_TASK_LEARNING.MASK.RATIO
        self.mask_threshold = self.cfg.ALGORITHM.MULTI_TASK_LEARNING.MASK.THRESHOLD
        self.mask_pre_amplitude = self.cfg.ALGORITHM.MULTI_TASK_LEARNING.MASK.PRE_AMPLITUDE
        self._reference_index = self.cfg.ALGORITHM.MULTI_TASK_LEARNING.MASK.REFERENCE_INDEX

        self.save_grad = config.ALGORITHM.MULTI_TASK_LEARNING.SAVE_GRAD
        self.save_grad_order = config.ALGORITHM.MULTI_TASK_LEARNING.SAVE_GRAD_ORDER
        assert self.save_grad_order+1 < 4, "only support 3 order gradient."

    def _get_params(self):
        return [self._params]

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self._get_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self._get_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count + 1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode):
        '''
        mode: backward, autograd
        '''

        grads = torch.empty(self.task_num, self.grad_dim).to(self.device)
        for tn in range(self.task_num):
            if mode == 'backward':
                if not self.save_grad:
                    losses[tn].backward(retain_graph=True) if (tn + 1) != self.task_num else losses[tn].backward()
                else:
                    losses[tn].backward(retain_graph=True)
                grads[tn] = self._grad2vec()
            elif mode == 'autograd': #todo: mode bug
                grad = list(torch.autograd.grad(losses[tn], self._get_params()[0], retain_graph=True))
                grads[tn] = torch.cat([g.view(-1) for g in grad])
            else:
                raise ValueError('No support {} mode for gradient computation')
            # very important!!!!
            if mode == "backward":
                self._get_params()[0].grad.data.zero_()

        return grads

    # compute once
    def _compute_n_order_grad(self, losses, n_order):

        grads = torch.zeros(n_order, self.task_num, self.grad_dim).to(losses.device)
        for tn in range(self.task_num):
            # very important!!!!
            self._get_params()[0].grad.data.zero_()
            for order in range(n_order):
                if order == n_order-1 and tn == self.task_num-1:
                    retain_graph = False
                    create_graph = False
                else:
                    retain_graph = True
                    create_graph = True
                if order == 0:
                    output = torch.autograd.grad(losses[tn], self._get_params()[0],
                                                 retain_graph=retain_graph,
                                                 create_graph=create_graph)
                else:
                    output = torch.autograd.grad(output[0].sum(), self._get_params()[0],
                                                 retain_graph=retain_graph,
                                                 create_graph=create_graph)
                grads[order, tn, :] = output[0].flatten()

        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self._get_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count + 1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1

    @property
    def reference_index(self):
        return int(self._reference_index) if self._reference_index is not None \
                    else self.task_num // 2 + (self.task_num // 2) // 2

    def _get_mask(self, grads, reference_index, mode="ratio", threshold=90, ratio=0.7):

        def norm(tensor):
            return tensor / (torch.norm(tensor))

        angles = torch.zeros(grads.shape[0], device=grads.device)

        for i, grad in enumerate(grads):

            # always maintain NOM corner
            if i == reference_index:
                angles[i] = 9999
                continue

            reference = norm(grads[reference_index])
            grad = norm(grad)
            angle = torch.acos(torch.clamp(torch.dot(grad, reference), -1., 1.)) * (180 / math.pi)
            angles[i] = angle

        if mode == "ratio":
            _, topk = torch.topk(angles, int(len(angles)*ratio))
            return topk
        else:
            return (angles>threshold).nonzero().squeeze()



    def _get_grads(self, losses, mode='backward'):
        r"""This function is used to return the gradients of representations or shared parameters.

        """
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode)
        return grads

    def _backward_new_grads(self, batch_weight, grads=None):
        r"""This function is used to reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].

        """
        # new_grads[new_grads>0]
        # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
        new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])

        self._reset_grad(new_grads)


    def save_grads(self, grads, dir, order, file):
        grads_show = grads.detach().clone()
        save_path = os.path.join("./grad_analysis", dir+'_'+'_'.join(self.cfg.OUTPUT_DIR.split('_')[1:]), order)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(grads_show, save_path + f'/{file}.pkl')


    @get_objectives_num
    def backward(self, losses=None, **kwargs):
        r"""
        Args:
            losses (list): A list of objectives of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        raise NotImplementedError("you should implement `backward` method in your class")