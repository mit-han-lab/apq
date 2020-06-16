# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

from imagenet_codebase.run_manager import ImagenetRunConfig, RunManager
import os
import copy
import torch
from elastic_nn.modules.dynamic_op import DynamicSeparableConv2d, DynamicSeparableQConv2d
from elastic_nn.networks.dynamic_quantized_proxyless import DynamicQuantizedProxylessNASNets
import json
import argparse

parser = argparse.ArgumentParser(description='Quantization-aware Finetuning')
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--id', type=int, default=-1)
args, _ = parser.parse_known_args()
print(args)

if __name__ == '__main__':
    exp_dir = 'exps/{}'.format(args.exp_name)

    arch_path = '{}/arch'.format(exp_dir)
    tmp_lst = json.load(open(arch_path, 'r'))
    info, q_info = tmp_lst
    print(info)
    print(q_info)

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

    tmp_dynamic_proxyless = copy.deepcopy(dynamic_proxyless)

    run_manager = RunManager(exp_dir, tmp_dynamic_proxyless, run_config, init=False)

    tmp_dynamic_proxyless.set_active_subnet(**info)
    tmp_dynamic_proxyless.set_quantization_policy(**q_info)

    run_manager.reset_running_statistics()
    acc = run_manager.finetune()

    acc_list = []
    acc_list.append((json.dumps(info), json.dumps(q_info), acc))
    output_dir = '{}/acc'.format(exp_dir)
    json.dump(acc_list, open(output_dir, 'w'))
    print('[Finished] Acc: {}'.format(acc))
