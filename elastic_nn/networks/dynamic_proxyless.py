import copy
import random

import numpy as np
import torch

from elastic_nn.modules.dynamic_layers import DynamicMBConvLayer, DynamicConvLayer, DynamicLinearLayer
from imagenet_codebase.modules.layers import ConvLayer, IdentityLayer, LinearLayer, MBInvertedConvLayer
from imagenet_codebase.networks.proxyless_nets import ProxylessNASNets, MobileInvertedResidualBlock, AvgMBConvStage
from imagenet_codebase.utils import make_divisible, int2list, list_weighted_sum, AverageMeter
import torch.nn.functional as F


class DynamicProxylessNASNets(ProxylessNASNets):

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0.1, base_stage_width=None,
                 width_mult_list=1.0, ks_list=3, expand_ratio_list=6, depth_list=4,
                 depth_ensemble_list=None, depth_ensemble_mode='avg'):

        self.width_mult_list = int2list(width_mult_list, 1)
        self.ks_list = int2list(ks_list, 1)
        self.expand_ratio_list = int2list(expand_ratio_list, 1)
        self.depth_list = int2list(depth_list, 1)

        self.depth_ensemble_list = depth_ensemble_list
        self.depth_ensemble_mode = depth_ensemble_mode

        self.width_mult_list.sort()
        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        if base_stage_width == 'v2':
            base_stage_width = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        elif base_stage_width == 'old':
            base_stage_width = [32, 16, 32, 40, 80, 96, 192, 320, 1280]
        else:
            # ProxylessNAS Stage Width
            base_stage_width = [32, 16, 24, 40, 80, 96, 192, 320, 1280]

        input_channel = [make_divisible(base_stage_width[0] * width_mult, 8) for width_mult in self.width_mult_list]
        first_block_width = [make_divisible(base_stage_width[1] * width_mult, 8) for width_mult in self.width_mult_list]
        last_channel = [
            make_divisible(base_stage_width[-1] * width_mult, 8) if width_mult > 1.0 else base_stage_width[-1]
            for width_mult in self.width_mult_list
        ]

        # first conv layer
        if len(input_channel) == 1:
            first_conv = ConvLayer(
                3, max(input_channel), kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
            )
        else:
            first_conv = DynamicConvLayer(
                in_channel_list=int2list(3, len(input_channel)), out_channel_list=input_channel, kernel_size=3,
                stride=2, act_func='relu6'
            )
        # first block
        if len(first_block_width) == 1:
            first_block_conv = MBInvertedConvLayer(
                in_channels=max(input_channel), out_channels=max(first_block_width), kernel_size=3, stride=1,
                expand_ratio=1, act_func='relu6',
            )
        else:
            first_block_conv = DynamicMBConvLayer(
                in_channel_list=input_channel, out_channel_list=first_block_width, kernel_size_list=3,
                expand_ratio_list=1, stride=1, act_func='relu6',
            )
        first_block = MobileInvertedResidualBlock(first_block_conv, None)

        input_channel = first_block_width

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1

        stride_stages = [2, 2, 2, 1, 2, 1]
        if depth_list is None:
            n_block_list = [2, 3, 4, 3, 3, 1]
            self.depth_list = [4]
        else:
            n_block_list = [max(self.depth_list)] * 5 + [1]

        width_list = []
        for base_width in base_stage_width[2:-1]:
            width = [make_divisible(base_width * width_mult, 8) for width_mult in self.width_mult_list]
            width_list.append(width)

        for width, n_block, s in zip(width_list, n_block_list, stride_stages):
            self.block_group_info.append(
                ([_block_index + i for i in range(n_block)], width)
            )
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1

                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=int2list(input_channel, 1), out_channel_list=int2list(output_channel, 1),
                    kernel_size_list=ks_list, expand_ratio_list=expand_ratio_list, stride=stride, act_func='relu6',
                )

                if stride == 1 and input_channel == output_channel:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None

                mb_inverted_block = MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

                blocks.append(mb_inverted_block)
                input_channel = output_channel
        # 1x1_conv before global average pooling
        if len(last_channel) == 1:
            feature_mix_layer = ConvLayer(
                max(input_channel), max(last_channel), kernel_size=1, use_bn=True, act_func='relu6',
            )
            classifier = LinearLayer(max(last_channel), n_classes, dropout_rate=dropout_rate)
        else:
            feature_mix_layer = DynamicConvLayer(
                in_channel_list=input_channel, out_channel_list=last_channel, kernel_size=1, stride=1, act_func='relu6',
            )
            classifier = DynamicLinearLayer(
                in_features_list=last_channel, out_features=n_classes, bias=True, dropout_rate=dropout_rate
            )

        super(DynamicProxylessNASNets, self).__init__(first_conv, blocks, feature_mix_layer, classifier)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [
            len(block_idx) for block_idx, _ in self.block_group_info
        ]

        if self.depth_ensemble_list is not None:
            self.depth_ensemble_list.sort()

    """ MyNetwork required methods """

    def forward(self, x):
        # first conv
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)

        # blocks
        for stage_id, (block_idx, _) in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]

            if self.depth_ensemble_list is not None and len(block_idx) == max(self.depth_list):
                experts = []
                for d, idx in enumerate(active_idx):
                    x = self.blocks[idx](x)
                    if (d + 1) in self.depth_ensemble_list:
                        experts.append(x)
                if len(experts) > 0:
                    ensemble_weights = self.get_depth_ensemble_weights(stage_id, len(experts))
                    x = list_weighted_sum(experts, ensemble_weights)
            else:
                for idx in active_idx:
                    x = self.blocks[idx](x)

        # feature_mix_layer
        x = self.feature_mix_layer(x)

        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.blocks[0].module_str + '\n'

        for stage_id, (block_idx, _) in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str + '\n'
        return _str

    @property
    def config(self):
        return '%s' % self

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    def load_weights_from_proxylessnas(self, proxyless_model_dict):
        model_dict = self.state_dict()
        for key in proxyless_model_dict:
            if key in model_dict:
                new_key = key
            elif '.bn.bn.' in key:
                new_key = key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in key:
                new_key = key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in key:
                new_key = key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in key:
                new_key = key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in key:
                new_key = key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in key:
                new_key = key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = proxyless_model_dict[key]
        self.load_state_dict(model_dict)

    def partial_load(self, a, b):
        x = np.flip(np.array(a.size()) - np.array(b.size()), 0)
        # print(x)
        y = np.zeros(2 * len(x))
        for i in range(len(x)):
            y[i * 2 + 1] = x[i]
        pp = tuple(y.astype(int))
        return F.pad(b, pp)

    def partial_load_dw(self, a, b):
        x = np.flip(np.array(a.size()) - np.array(b.size()), 0)
        # print(x)
        y = np.zeros(2 * len(x))
        for i in range(len(x)):
            if i <= 1:
                assert x[i] % 2 == 0
                y[i * 2 + 1] = x[i] // 2
                y[i * 2] = x[i] // 2
            else:
                y[i * 2 + 1] = x[i]
        # print(y)
        pp = tuple(y.astype(int))
        return F.pad(b, pp)

    def load_partial_weights_from_proxylessnas(self, proxyless_model_dict):
        model_dict = self.state_dict()
        for key in proxyless_model_dict:
            if key in model_dict:
                new_key = key
            elif '.bn.bn.' in key:
                new_key = key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in key:
                new_key = key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in key:
                new_key = key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in key:
                new_key = key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in key:
                new_key = key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in key:
                new_key = key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(key)
            assert new_key in model_dict, '%s' % new_key
            # model_dict[new_key] = proxyless_model_dict[key]
            # if 'running' in new_key:
            #     continue
            if 'depth_conv.conv.weight' in key:
                # print(new_key, key)
                tmp = model_dict[new_key]
                tmp2 = proxyless_model_dict[key]
                # print(tmp.size())
                # print(tmp2.size())
                model_dict[new_key] = self.partial_load_dw(tmp, tmp2)
            else:
                # print(new_key, key)
                tmp = model_dict[new_key]
                tmp2 = proxyless_model_dict[key]
                # print(tmp.size())
                # print(tmp2.size())
                model_dict[new_key] = self.partial_load(tmp, tmp2)
            # print()
        self.load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_active_subnet(self, wid=None, ks=None, e=None, d=None):
        width_mult_id = int2list(wid, 3 + len(self.blocks) - 1)
        ks = int2list(ks, len(self.blocks) - 1)
        expand_ratio = int2list(e, len(self.blocks) - 1)
        depth = int2list(d, len(self.block_group_info))

        if len(self.width_mult_list) > 1 and width_mult_id[0] is not None:
            # active_out_channel
            self.first_conv.active_out_channel = self.first_conv.out_channel_list[width_mult_id[0]]
            self.blocks[0].mobile_inverted_conv.active_out_channel = \
                self.blocks[0].mobile_inverted_conv.out_channel_list[width_mult_id[1]]
            self.feature_mix_layer.active_out_channel = self.feature_mix_layer.out_channel_list[width_mult_id[2]]

        for block, w, k, e in zip(self.blocks[1:], width_mult_id[3:], ks, expand_ratio):
            if w is not None:
                block.mobile_inverted_conv.active_out_channel = block.mobile_inverted_conv.out_channel_list[w]
            if k is not None:
                block.mobile_inverted_conv.active_kernel_size = k
            if e is not None:
                block.mobile_inverted_conv.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i][0]), d)

    def sample_active_subnet(self):
        width_mult_candidates = [i for i in range(len(self.width_mult_list))]
        ks_candidates = self.ks_list
        expand_candidates = self.expand_ratio_list
        depth_candidates = self.depth_list

        # sample width_mult
        if len(self.width_mult_list) == 1:
            width_mult_setting = None
        else:
            width_mult_setting = random.choices(width_mult_candidates, k=3 + len(self.blocks) - 1)

        # sample kernel size
        ks_setting = random.choices(ks_candidates, k=len(self.blocks) - 1)

        # sample expand ratio
        # expand_setting = random.choices(expand_candidates, k=len(self.blocks) - 1)
        # expand_setting = random.choices(expand_candidates, k=len(self.blocks) - 1)
        expand_setting = (np.random.rand(21) * 2 + 4).tolist()
        assert expand_candidates == [4, 6]
        # print(np.random.rand(len(self.blocks) - 1).tolist())
        # print(expand_setting, expand_candidates)
        # sample depth
        depth_setting = random.choices(depth_candidates, k=len(self.block_group_info))

        self.set_active_subnet(width_mult_setting, ks_setting, expand_setting, depth_setting)

        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        if len(self.width_mult_list) == 1:
            first_conv = copy.deepcopy(self.first_conv)
            blocks = [copy.deepcopy(self.blocks[0])]
            feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
            classifier = copy.deepcopy(self.classifier)
        else:
            first_conv = self.first_conv.get_active_subnet(3, preserve_weight)
            blocks = [MobileInvertedResidualBlock(
                self.blocks[0].mobile_inverted_conv.get_active_subnet(first_conv.out_channels, preserve_weight),
                copy.deepcopy(self.blocks[0].shortcut)
            )]
            feature_mix_layer = self.feature_mix_layer.get_active_subnet(
                self.blocks[-1].mobile_inverted_conv.active_out_channel, preserve_weight)
            classifier = self.classifier.get_active_subnet(self.feature_mix_layer.active_out_channel, preserve_weight)

        input_channel = blocks[0].mobile_inverted_conv.out_channels
        # blocks
        for stage_id, (block_idx, _) in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(MobileInvertedResidualBlock(
                    self.blocks[idx].mobile_inverted_conv.get_active_subnet(input_channel, preserve_weight),
                    copy.deepcopy(self.blocks[idx].shortcut)
                ))
                input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
            if self.depth_ensemble_list is None:
                blocks += stage_blocks
            else:
                assert self.depth_ensemble_mode == 'avg'
                if len(stage_blocks) == 3:
                    stage_blocks[-1].mobile_inverted_conv.point_linear.bn.weight.data.mul_(0.5)
                    stage_blocks[-1].mobile_inverted_conv.point_linear.bn.bias.data.mul_(0.5)
                    blocks += stage_blocks
                elif len(stage_blocks) == 4:
                    stage_blocks[-1].mobile_inverted_conv.point_linear.bn.weight.data.mul_(1 / 3)
                    stage_blocks[-1].mobile_inverted_conv.point_linear.bn.bias.data.mul_(1 / 3)
                    mbconv_stage = AvgMBConvStage(stage_blocks)
                    blocks.append(mbconv_stage)
                else:
                    blocks += stage_blocks

        active_subnet = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        active_subnet.set_bn_param(**self.get_bn_param())
        return active_subnet

    """ Depth Related Methods """

    def get_depth_ensemble_weights(self, stage_id, n_experts=None):
        if n_experts is None:
            n_experts = len(self.depth_ensemble_list)

        if self.depth_ensemble_mode == 'avg':
            return [1.0 / n_experts for _ in range(n_experts)]
        else:
            raise NotImplementedError
