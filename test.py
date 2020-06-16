# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.latency_predictor import LatencyPredictor
import sys
import copy
import argparse
import os
import json
import torch
from elastic_nn.modules.dynamic_op import DynamicSeparableConv2d, DynamicSeparableQConv2d
from elastic_nn.networks.dynamic_quantized_proxyless import DynamicQuantizedProxylessNASNets
from imagenet_codebase.run_manager import ImagenetRunConfig, RunManager

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--exp_dir', type=str, default=None)
args, _ = parser.parse_known_args()


if __name__ == '__main__':
    latency_predictor = LatencyPredictor(type='latency')
    energy_predictor = LatencyPredictor(type='energy')
    arch_dir = '{}/arch'.format(args.exp_dir)
    assert os.path.exists(arch_dir)
    tmp_lst = json.load(open(arch_dir, 'r'))
    info, q_info = tmp_lst
    print(info)
    print(q_info)

    X = LatencyPredictor(type='latency')
    print('Latency: {:.2f}ms'.format(X.predict_lat(dict(info, **q_info))))

    Y = LatencyPredictor(type='energy')
    print('Energy: {:.2f}mJ'.format(Y.predict_lat(dict(info, **q_info))))
    ckpt_path = '{}/checkpoint/model_best.pth.tar'.format(args.exp_dir)
    if os.path.exists(ckpt_path):
        DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1
        DynamicSeparableQConv2d.KERNEL_TRANSFORM_MODE = 1

        dynamic_proxyless = DynamicQuantizedProxylessNASNets(
            ks_list=[3, 5, 7], expand_ratio_list=[4, 6], depth_list=[2, 3, 4], base_stage_width='proxyless',
            width_mult_list=1.0, dropout_rate=0, n_classes=1000
        )

        proxylessnas_init = torch.load(
            './models/imagenet-OFA',
            map_location='cpu'
        )['state_dict']
        dynamic_proxyless.load_weights_from_proxylessnas(proxylessnas_init)
        init_lr = 1e-3
        run_config = ImagenetRunConfig(
            test_batch_size=1000, image_size=224, n_worker=16, valid_size=5000, dataset='imagenet', train_batch_size=256,
            init_lr=init_lr, n_epochs=30,
        )

        run_manager = RunManager('~/tmp', dynamic_proxyless, run_config, init=False)

        proxylessnas_init = torch.load(
            ckpt_path,
            map_location='cpu'
        )['state_dict']
        dynamic_proxyless.load_weights_from_proxylessnas(proxylessnas_init)

        dynamic_proxyless.set_active_subnet(**info)
        dynamic_proxyless.set_quantization_policy(**q_info)

        acc = run_manager.validate(is_test=True)
        print('Accuracy: {:.1f}'.format(acc[1]))
