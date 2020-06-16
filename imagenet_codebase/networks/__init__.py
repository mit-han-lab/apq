# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

from imagenet_codebase.networks.proxyless_nets import ProxylessNASNets, proxyless_base


def get_net_by_name(name):
    if name == ProxylessNASNets.__name__:
        return ProxylessNASNets
    else:
        raise ValueError('unrecognized type of network: %s' % name)
