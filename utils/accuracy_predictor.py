# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

import torch
import os
import torch.nn as nn
import numpy as np
from utils.converter import Converter

cvt = Converter()


def preparation(quantize=False, file_name=None, all=False, data_size=None):
    from utils.converter import Converter

    def data_loader(file_name='dataset/NetInfo/acc/info.dict', quantize=False):
        import json

        lst = json.load(open(file_name, 'r'))
        X_all = []
        y_all = []
        converter = Converter()
        for k, v in lst.items():
            dic = json.loads(k)
            tmp = converter.spec2feature(dic, quantize)
            X_all.append(tmp)
            y_all.append(v / 100.)
        return X_all, y_all

    if file_name is None:
        file_name = 'dataset/NetInfo/acc_quant/info.dict' if quantize else 'dataset/NetInfo/acc/info.dict'
    X_all, y_all = data_loader(file_name, quantize)

    X_all = torch.tensor(X_all, dtype=torch.float)
    y_all = torch.tensor(y_all)

    if not quantize:
        X_all = X_all[:, :cvt.half_dim]

    y_all = y_all
    shuffle_idx = torch.randperm(len(X_all))
    X_all = X_all[shuffle_idx]
    y_all = y_all[shuffle_idx]

    if all:
        X_train = X_all
        y_train = y_all
        X_test = X_all
        y_test = y_all
    elif data_size is not None:
        X_train = X_all[:data_size, :]
        y_train = y_all[:data_size]
        X_test = None
        y_test = None
    else:
        idx = X_all.size(0) // 5 * 4
        X_train = X_all[:idx]
        y_train = y_all[:idx]
        X_test = X_all[idx:]
        y_test = y_all[idx:]

    return X_train, y_train, X_test, y_test


class MLP(nn.Module):
    def __init__(self, args, pretrained=True, mlp_hidden_size=400, mlp_layers=3, quantize=False, scratch=False):
        super(MLP, self).__init__()
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_layers = mlp_layers
        self.quantize = quantize
        self.layers = nn.ModuleList()

        for i in range(self.mlp_layers):
            if i == 0:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(cvt.half_dim, self.mlp_hidden_size),
                        nn.ReLU(inplace=False),
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                        nn.ReLU(inplace=False),
                    )
                )
        if self.quantize:
            self.quantize_fc = nn.Linear(cvt.full_dim - cvt.half_dim, self.mlp_hidden_size)
        self.regressor = nn.Linear(self.mlp_hidden_size, 1)

        if pretrained:
            self.save_path = '{}/acc.pt'.format(args.acc_predictor_dir) if not self.quantize \
                else '{}/acc_quant.pt'.format(args.acc_predictor_dir)
            print('Load from {}'.format(self.save_path))
            self.update_acc_state_dict()
        elif quantize and not pretrained:
            self.save_path = '{}/acc.pt'.format(args.acc_predictor_dir)
            if not scratch:
                self.update_acc_state_dict(strict=False)

    def update_acc_state_dict(self, strict=True):
        if not os.path.exists(self.save_path):
            assert False
        self.load_state_dict(torch.load(self.save_path), strict=strict)

    def forward(self, x):
        if self.quantize:
            x1, x2 = x[:, :cvt.half_dim], x[:, cvt.half_dim:]
            x = self.layers[0](x1) + self.quantize_fc(x2)
        else:
            x = self.layers[0](x)

        for i in range(1, self.mlp_layers):
            x = self.layers[i](x)

        x = self.regressor(x)
        return x


class AccuracyPredictor():
    def __init__(self, args, quantize=True):
        assert quantize
        self.quantize = quantize
        if self.quantize:
            self.mlp_with_q = MLP(args, quantize=True).cuda()
            self.mlp_with_q.eval()

    def predict_accuracy(self, specs):
        assert self.quantize
        X = []
        for spec in specs:
            X.append(cvt.spec2feature(spec))
        X = np.array(X)
        X = torch.tensor(X).float().cuda()
        if self.quantize:
            y = self.mlp_with_q(X)
        else:
            y = self.mlp(X[:, :cvt.half_dim])
        return y


if __name__ == "__main__":
    mlp = MLP(pretrained=False)
    eval(mlp)
