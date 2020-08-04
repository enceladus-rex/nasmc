import torch

from typing import Tuple


def batch_apply_rnn(
        inputs: torch.Tensor, cell_state: Tuple[torch.Tensor, torch.Tensor],
        cell: torch.nn.RNNCellBase) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_dims = inputs.shape[:-1]
    h, c = cell_state

    reshaped_inputs = inputs.reshape((-1, inputs.shape[-1]))
    reshaped_h = h.reshape((-1, h.shape[-1]))
    reshaped_c = c.reshape((-1, c.shape[-1]))

    hnext, cnext = cell(reshaped_inputs, (reshaped_h, reshaped_c))

    hnext = hnext.reshape(batch_dims + (hnext.shape[-1], ))
    cnext = cnext.reshape(batch_dims + (cnext.shape[-1], ))

    return (hnext, cnext)
