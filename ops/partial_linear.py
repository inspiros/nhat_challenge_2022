import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'PartialLinear'
]


class PartialLinear(nn.Linear):

    def __init__(self, *args, **kwargs):
        self.return_mask = kwargs.pop('return_mask', True)
        self.eps = kwargs.pop('eps', 1e-8)
        super(PartialLinear, self).__init__(*args, **kwargs)

        self.register_buffer(
            'mask_update_weight',
            torch.ones_like(self.weight)
        )

    def forward(self, x, mask=None):
        output = F.linear(x, self.weight)
        with torch.no_grad():
            mask = F.linear(mask, self.mask_update_weight)
            mask = mask / self.in_features
            mask_ratio = 1 / (mask + self.eps)
            mask = mask.bool().to(x.dtype)
            mask_ratio = mask_ratio * mask

        if self.bias is not None:
            bias_view = self.bias.view(1, -1, 1, 1)
            output = mask * (output * mask_ratio + bias_view)
        else:
            output = output * mask_ratio

        if self.return_mask:
            return output, mask
        return output
