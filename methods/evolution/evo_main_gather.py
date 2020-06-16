# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.accuracy_predictor import AccuracyPredictor
from utils.latency_predictor import LatencyPredictor
from methods.evolution.evolution_finder import EvolutionFinder
import argparse


def evolution_gather(parser: argparse.ArgumentParser, force_latency):
    parser.add_argument('--acc_predictor_dir', type=str, default='./models')
    parser.add_argument('--type', type=str, default='latency')
    args = parser.parse_args()
    accuracy_predictor = AccuracyPredictor(args, quantize=True)
    latency_predictor = LatencyPredictor(type=args.type)
    t = 500
    evolution_finder = EvolutionFinder(latency_predictor, accuracy_predictor)
    times, best_valid, info = evolution_finder.run_evolution_search(constraint=force_latency, max_time_budget=t)
    print(info)
    return best_valid[-1], info, t


if __name__ == '__main__':
    pass
