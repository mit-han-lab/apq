# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

import json

from imagenet_codebase.modules.layers import *


def proxyless_base(net_config=None, n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0,
                   local_path='~/.torch/proxylessnas/'):
    assert net_config is not None, 'Please input a network config'
    if 'http' in net_config:
        net_config_path = download_url(net_config, local_path)
    else:
        net_config_path = net_config
    net_config_json = json.load(open(net_config_path, 'r'))

    net_config_json['classifier']['out_features'] = n_classes
    net_config_json['classifier']['dropout_rate'] = dropout_rate

    net = ProxylessNASNets.build_from_config(net_config_json)
    if bn_param is not None:
        net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    return net


class AvgMBConvStage(MyModule):

    def __init__(self, mb_conv_list):
        super(AvgMBConvStage, self).__init__()

        assert len(mb_conv_list) == 4
        self.mb_conv_list = nn.ModuleList(mb_conv_list)

    def forward(self, x):
        for mb_conv in self.mb_conv_list[:2]:
            x = mb_conv(x)
        f2_o2 = self.mb_conv_list[2].mobile_inverted_conv(x)
        stage_out = 2 / 3 * f2_o2 + x
        o3 = f2_o2 + x
        return stage_out + self.mb_conv_list[3].mobile_inverted_conv(o3)

    @property
    def module_str(self):
        _str = []
        for mb_conv in self.mb_conv_list:
            _str.append(mb_conv.module_str)
        return '\n'.join(_str)

    @property
    def config(self):
        return {
            'name': AvgMBConvStage.__name__,
            'mb_conv_list': [
                mb_conv.config for mb_conv in self.mb_conv_list
            ]
        }

    @staticmethod
    def build_from_config(config):
        mb_conv_list = []
        for mb_conv_config in config['mb_conv_list']:
            mb_conv_list.append(MobileInvertedResidualBlock.build_from_config(mb_conv_config))
        return AvgMBConvStage(mb_conv_list)


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.mobile_inverted_conv(x)
        else:
            res = self.mobile_inverted_conv(x) + self.shortcut(x)
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str if self.mobile_inverted_conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config if self.mobile_inverted_conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)


class MobileInvertedResidualQBlock(QModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualQBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.mobile_inverted_conv(x)
        else:
            res = self.mobile_inverted_conv(x) + self.shortcut(x)
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str if self.mobile_inverted_conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config if self.mobile_inverted_conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)


class ProxylessNASNets(MyNetwork):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNets, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.module_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': None if self.feature_mix_layer is None else self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_config in config['blocks']:
            if block_config['name'] == AvgMBConvStage.__name__:
                blocks.append(AvgMBConvStage.build_from_config(block_config))
            else:
                blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net
