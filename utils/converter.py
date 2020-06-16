# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

import random
import numpy as np


class Converter():
    def __init__(self):
        self.cnt = 0
        self.k_info = dict(id2val=[], val2id=[], L=[], R=[])
        self.e_info = dict(id2val=[], val2id=[], L=[], R=[])
        self.q_info = [dict(id2val=[], val2id=[], L=[], R=[]) for it in range(4)]  # w_pw a_pw w_dw a_dw

        self.num2qname = {0: 'pw_w_bits_setting', 1: 'pw_a_bits_setting', 2: 'dw_w_bits_setting',
                          3: 'dw_a_bits_setting'}
        self.build(self.k_info, [3, 5, 7])
        self.build(self.e_info)
        self.half_dim = self.cnt

        for it in range(4):
            self.build(self.q_info[it], [4, 6, 8])
        self.full_dim = self.cnt

    def build(self, info_dic, ls=None):
        if ls is None:
            lst = []
            lst.extend([16])
            lst.extend([24] * 4)
            lst.extend([40] * 4)
            lst.extend([80] * 4)
            lst.extend([96] * 4)
            lst.extend([192] * 4)
            for i in range(21):
                t = lst[i]
                dic = {}
                dic2 = {}
                info_dic['L'].append(self.cnt)
                for k in range(t * 4, t * 6 + 8, 8):
                    dic[k / t] = self.cnt
                    dic2[self.cnt] = k / t
                    self.cnt += 1
                info_dic['R'].append(self.cnt)
                info_dic['val2id'].append(dic)
                info_dic['id2val'].append(dic2)
        else:
            for i in range(21):
                dic = {}
                dic2 = {}
                info_dic['L'].append(self.cnt)
                for k in ls:
                    dic[k] = self.cnt
                    dic2[self.cnt] = k
                    self.cnt += 1
                info_dic['R'].append(self.cnt)
                info_dic['val2id'].append(dic)
                info_dic['id2val'].append(dic2)

    def spec2feature(self, info, quantize=False):
        def make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            x = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # make sure that round down does not go down by more than 10%
            if x < 0.9 * v:
                x += divisor
            return x

        ks = info['ks']
        e = info['e']
        d = info['d']
        if not quantize:
            q = [[8] * 21 for _ in range(4)]
        else:
            q = []
            for it in range(4):
                q.append(info[self.num2qname[it]])
        # qa = [32] * 21

        spec = np.zeros(self.cnt)
        for i in range(21):
            nowd = i % 4
            stg = i // 4
            if nowd < d[stg]:
                spec[self.k_info['val2id'][i][ks[i]]] = 1

        channel_in_list = [16, 24, 40, 80, 96, 192]
        channel_out_list = [24, 40, 80, 96, 192, 320]

        for i in range(21):
            inc = channel_in_list[i // 4] if i % 4 == 0 else channel_in_list[i // 4 + 1]
            ouc = channel_out_list[i // 4]
            nowd = i % 4
            stg = i // 4
            if nowd < d[stg]:
                if e[i] in self.e_info['val2id'][i]:
                    spec[self.e_info['val2id'][i][e[i]]] = 1
                else:
                    real_e = make_divisible(e[i] * inc, 8) / inc
                    spec[self.e_info['val2id'][i][real_e]] = 1
                    assert min(self.e_info['val2id'][i].keys(), key=lambda key: abs(key - e[i])) == real_e

        for it in range(4):
            for i in range(21):
                nowd = i % 4
                stg = i // 4
                if nowd < d[stg]:
                    spec[self.q_info[it]['val2id'][i][q[it][i]]] = 1

        return spec

    def feature2spec(self, spec):
        info = {'wid': None, 'ks': [],
                'e': [], 'd': [], 'pw_w_bits_setting': [], 'pw_a_bits_setting': [], 'dw_w_bits_setting': [],
                'dw_a_bits_setting': []}
        d = 0
        for i in range(21):
            identity = True
            for j in range(self.k_info['L'][i], self.k_info['R'][i]):
                if spec[j] == 1:
                    info['ks'].append(self.k_info['id2val'][i][j])
                    identity = False
                    break

            for j in range(self.e_info['L'][i], self.e_info['R'][i]):
                if spec[j] == 1:
                    info['e'].append(self.e_info['id2val'][i][j])
                    identity = False
                    break

            for it in range(4):
                for j in range(self.q_info[it]['L'][i], self.q_info[it]['R'][i]):
                    if spec[j] == 1:
                        info[self.num2qname[it]].append(self.q_info[it]['id2val'][i][j])

            if identity:
                info['e'].append(4)
                info['ks'].append(3)
                for it in range(4):
                    info[self.num2qname[it]].append(8)
            else:
                d += 1

            if i % 4 == 3:
                info['d'].append(d)
                d = 0

        info['d'].append(d)
        return info

    def is_valid(self, spec):
        for i in range(21):
            if spec['ks'][i] == 7 and spec['dw_a_bits_setting'][i] == 4:
                return False
        return True

    def random_spec(self):
        info = {'wid': None, 'ks': [],
                'e': [], 'd': [], 'pw_w_bits_setting': [], 'pw_a_bits_setting': [], 'dw_w_bits_setting': [],
                'dw_a_bits_setting': []}

        for i in range(6):
            info['d'].append(np.random.randint(3) + 2)

        for i in range(21):
            info['ks'].append(random.sample(self.k_info['val2id'][i].keys(), 1)[0])
            info['e'].append(random.sample(self.e_info['val2id'][i].keys(), 1)[0])
            for it in range(4):
                info[self.num2qname[it]].append(random.sample(self.q_info[it]['val2id'][i].keys(), 1)[0])

        return info

    def change_spec(self, spec, i):
        spec['ks'][i] = random.sample(self.k_info['val2id'][i].keys(), 1)[0]
        spec['e'][i] = random.sample(self.e_info['val2id'][i].keys(), 1)[0]
        for it in range(4):
            spec[self.num2qname[it]][i] = random.sample(self.q_info[it]['val2id'][i].keys(), 1)[0]

        return spec


if __name__ == '__main__':
    test()
