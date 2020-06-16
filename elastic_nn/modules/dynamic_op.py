import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch

from imagenet_codebase.utils import get_same_padding, sub_filter_start_end
from imagenet_codebase.utils import QModule


class DynamicSeparableConv2d(nn.Module):
    KERNEL_TRANSFORM_MODE = None  # None or 1
    
    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1):
        super(DynamicSeparableConv2d, self).__init__()
        
        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation
        
        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_in_channels, max(self.kernel_size_list), self.stride,
            groups=self.max_in_channels, bias=False,
        )
        
        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)
    
    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)
        
        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters
    
    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)
        
        filters = self.get_active_filter(in_channel, kernel_size).contiguous()
        
        padding = get_same_padding(kernel_size)
        y = F.conv2d(
            x, filters, None, self.stride, padding, self.dilation, in_channel
        )
        return y


class DynamicSeparableQConv2d(QModule):
    KERNEL_TRANSFORM_MODE = None  # None or 1

    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1,
                 w_bit=-1, a_bit=-1, half_wave=True):
        super(DynamicSeparableQConv2d, self).__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave)

        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_in_channels, max(self.kernel_size_list), self.stride,
            groups=self.max_in_channels, bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)

        filters = self.get_active_filter(in_channel, kernel_size).contiguous()
        inputs, weight, bias = self._quantize(inputs=x, weight=filters, bias=None)

        padding = get_same_padding(kernel_size)
        y = F.conv2d(
            inputs, weight, None, self.stride, padding, self.dilation, in_channel
        )
        return y


class DynamicPointConv2d(nn.Module):

    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1):
        super(DynamicPointConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.conv.weight[:out_channel, :in_channel, :, :].contiguous()

        padding = get_same_padding(self.kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicPointQConv2d(QModule):
    
    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1,
                 w_bit=-1, a_bit=-1, half_wave=True):
        super(DynamicPointQConv2d, self).__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave)

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
        )
        
        self.active_out_channel = self.max_out_channels
    
    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.conv.weight[:out_channel, :in_channel, :, :].contiguous()
        inputs, weight, bias = self._quantize(inputs=x, weight=filters, bias=None)

        padding = get_same_padding(self.kernel_size)
        y = F.conv2d(inputs, weight, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicLinear(nn.Module):
    
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()
        
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias
        
        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)
        
        self.active_out_features = self.max_out_features
    
    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features
        
        in_features = x.size(1)
        weight = self.linear.weight[:out_features, :in_features]
        bias = self.linear.bias[:out_features] if self.bias else None
        y = F.linear(x, weight, bias)
        return y


class DynamicQLinear(QModule):

    def __init__(self, max_in_features, max_out_features, bias=True,
                 w_bit=-1, a_bit=-1, half_wave=True):
        super(DynamicQLinear, self).__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave)

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.linear.weight[:out_features, :in_features]
        bias = self.linear.bias[:out_features] if self.bias else None
        inputs, weight, bias = self._quantize(inputs=x, weight=weight, bias=bias)
        y = F.linear(inputs, weight, bias)
        return y


class DynamicBatchNorm2d(nn.Module):
    SET_RUNNING_STATISTICS = False
    
    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()
        
        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)
    
    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0
            
            if bn.training and bn.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x, bn.running_mean[:feature_dim], bn.running_var[:feature_dim], bn.weight[:feature_dim],
                bn.bias[:feature_dim], bn.training or not bn.track_running_stats,
                exponential_average_factor, bn.eps,
            )
    
    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y
