# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

import argparse
import json

from methods.evolution.evo_main_gather import evolution_gather

parser = argparse.ArgumentParser(description='Best Arch Searcher')
parser.add_argument('--prepare', type=str, default=None, choices=['acc', 'acc_quant'])
parser.add_argument('--acc_train_sample', type=int, default=None)
parser.add_argument('--mode', type=str, default='evolution', choices=['evolution'])
parser.add_argument('--constraint', type=float, default=120)
parser.add_argument('--exp_name', type=str, default='test')
args, _ = parser.parse_known_args()
print(args)


def main():
    import copy
    import os
    if args.mode == 'evolution':
        def add_arch(info, lst):
            info1 = copy.deepcopy(info)
            info2 = copy.deepcopy(info)
            del info1['dw_w_bits_setting']
            del info1['dw_a_bits_setting']
            del info1['pw_w_bits_setting']
            del info1['pw_a_bits_setting']
            del info2['wid']
            del info2['ks']
            del info2['e']
            del info2['d']
            lst.append((info1, info2))

        dic = {}
        whole = {}
        candidate_archs = []
        out_dir = 'exps/{}'.format(args.exp_name)
        lats = []
        for i in [args.constraint]:
            res, info, t = evolution_gather(parser, force_latency=i)
            acc, arch, lat = info
            print((i, res, lat, arch, acc))
            if i not in dic or dic[i] < acc:
                dic[i] = acc
                whole[i] = (t, res, lat, arch, acc)
        lats.append(lat)
        add_arch(arch, candidate_archs)
        print('Found Best Architecture: {}'.format(dic))
        os.makedirs(out_dir, exist_ok=True)
        json.dump(candidate_archs[0], open('{}/arch'.format(out_dir), 'w'))
        json.dump(lats, open('{}/lat'.format(out_dir), 'w'))


if __name__ == '__main__':
    main()
