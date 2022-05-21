import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['PartialConv2d', 'PartialConvTranspose2d']


# noinspection DuplicatedCode
class PartialConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        self.multi_channel = kwargs.pop('multi_channel', True)
        self.return_mask = kwargs.pop('return_mask', True)
        self.eps = kwargs.pop('eps', 1e-8)
        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for PartialConv2d')

        if self.multi_channel:
            mask_update_weight = torch.ones(self.out_channels,
                                            self.in_channels,
                                            *self.kernel_size)
            self.slide_winsize = self.in_channels * np.prod(self.kernel_size)
        else:
            mask_update_weight = torch.ones(1, 1, *self.kernel_size)
            self.slide_winsize = np.prod(self.kernel_size)
        self.register_buffer('mask_update_weight', mask_update_weight)

    def forward(self, input, mask=None):
        n, c, h, w = input.shape

        with torch.no_grad():
            mask_shape = (n if self.multi_channel else 1,
                          c if self.multi_channel else 1,
                          h, w)
            if mask is None:
                mask = torch.ones(*mask_shape).to(input)
            else:
                if mask.shape != mask_shape:
                    raise ValueError(f'mask must be tensor of shape {mask_shape},'
                                     f' but got {tuple(mask.shape)}.')

            out_mask = F.conv2d(mask, self.mask_update_weight, None,
                                self.stride, self.padding, self.dilation, 1)
            out_mask = out_mask / self.slide_winsize
            mask_ratio = 1 / (out_mask + self.eps)
            out_mask = out_mask.bool().to(input.dtype)
            mask_ratio *= out_mask

        output = F.conv2d(input * mask,
                          self.weight, None, self.stride, self.padding,
                          self.dilation, self.groups)

        if self.bias is not None:
            bias_view = self.bias.view(1, -1, 1, 1)
            output = out_mask * (output * mask_ratio + bias_view)
        else:
            output = output * mask_ratio

        if self.return_mask:
            return output, out_mask
        return output


# noinspection DuplicatedCode
class PartialConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        self.multi_channel = kwargs.pop('multi_channel', False)
        self.return_mask = kwargs.pop('return_mask', False)
        self.eps = kwargs.pop('eps', 1e-8)
        super(PartialConvTranspose2d, self).__init__(*args, **kwargs)

        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for PartialConvTransposed2d')

        if self.multi_channel:
            self.mask_update_weight = torch.ones(self.in_channels,
                                                 self.out_channels,
                                                 *self.kernel_size)
            self.slide_winsize = self.out_channels * np.prod(self.kernel_size)
        else:
            self.mask_update_weight = torch.ones(1, 1, *self.kernel_size)
            self.slide_winsize = np.prod(self.kernel_size)

    def forward(self, input, mask=None, output_size=None):
        n, c, h, w = input.shape
        output_padding = self._output_padding(
            input, output_size,
            self.stride, self.padding,  # type: ignore[arg-type]
            self.kernel_size, self.dilation)  # type: ignore[arg-type]

        with torch.no_grad():
            mask_shape = (n if self.multi_channel else 1,
                          c if self.multi_channel else 1,
                          h, w)
            if mask is None:
                mask = torch.ones(*mask_shape).to(input)
            else:
                if mask.shape != mask_shape:
                    raise ValueError(f'mask must be tensor of shape {mask_shape},'
                                     f' but got {tuple(mask.shape)}.')

            out_mask = F.conv_transpose2d(mask, self.mask_update_weight, bias=None, stride=self.stride,
                                          padding=self.padding, output_padding=output_padding,
                                          dilation=self.dilation, groups=1)
            out_mask = out_mask / self.slide_winsize
            mask_ratio = 1 / (out_mask + self.eps)
            out_mask = out_mask.bool().to(input.dtype)
            mask_ratio *= out_mask

        output = F.conv_transpose2d(input * mask,
                                    self.weight, None, self.stride, self.padding,
                                    output_padding, self.groups, self.dilation)

        if self.bias is not None:
            bias_view = self.bias.view(1, -1, 1, 1)
            output = out_mask * (output * mask_ratio + bias_view)
        else:
            output = output * mask_ratio

        if self.return_mask:
            return output, out_mask
        return output
