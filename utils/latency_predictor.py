# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

import json

class LatencyPredictor():
    def __init__(self, platform='BitFusion', type='latency', batch=1):
        for input_size in range(224, 224 + 4, 4):
            self.input_size = input_size
            self.sz_in = [self.input_size]
            self.sz_out = [(self.input_size + 1) // 2]
            for i in range(6):
                self.sz_in.append(self.sz_out[-1])
                if i == 3 or i == 5:
                    self.sz_out.append(self.sz_in[-1])
                else:
                    self.sz_out.append((self.sz_in[-1] + 1) // 2)

        assert platform in ['BitFusion']
        assert type in ['latency', 'energy']
        self.platform = platform
        self.type = type
        file_path = 'lut/{}_new.b{}.dict'.format(self.platform, batch)  # bitfusion batch=16
        self.dic = json.load(open(file_path, 'r'))
        self.other = self.dic['head'][self.type] + self.dic['tail'][self.type]

    def build_table(self):
        def add(tmp, dic, measure_table):
            if tmp not in dic:
                dic[tmp] = 1
                measure_table.append(tmp)

        channel_in_list = [16, 24, 40, 80, 96, 192]
        channel_out_list = [24, 40, 80, 96, 192, 320]
        input_list = []
        output_list = []
        for i in range(1, 7):
            input_list.append('{}x{}x{}'.format(self.sz_in[i], self.sz_in[i], channel_in_list[i - 1]))
            output_list.append('{}x{}x{}'.format(self.sz_out[i], self.sz_out[i], channel_out_list[i - 1]))

        kernel_list = [3, 5, 7]
        dic = {}
        measure_table = []
        for layer in range(21):
            inp = input_list[layer // 4] if layer % 4 == 0 else input_list[layer // 4 + 1]
            out = output_list[layer // 4]
            input_sz = int(inp.split('x')[0])
            output_sz = int(out.split('x')[0])
            in_channels = int(inp.split('x')[-1])
            out_channels = int(out.split('x')[-1])
            stride = 1 if input_sz == output_sz else 2
            idskip = 1 if stride == 1 and in_channels == out_channels else 0
            for ks in kernel_list:
                lst = list(range(in_channels * 4, in_channels * 6 + 8, 8))
                for mid_c in lst:
                    out_c = out_channels
                    # in_channels, mid_channels, out_channels, input_size, kernel_size, stride, idskip
                    tmp = in_channels, mid_c, input_sz, 1, 1, 0, 1  # pw
                    add(tmp, dic, measure_table)
                    tmp = mid_c, mid_c, input_sz, ks, stride, ks // 2, mid_c  # dw
                    add(tmp, dic, measure_table)
                    tmp = mid_c, out_c, input_sz // stride, 1, 1, 0, 1  # pw
                    add(tmp, dic, measure_table)
        return measure_table

    def get_lat(self, tmp, q):
        info = '{}-W{}A{}'.format(tmp, q[0], q[1])
        # print(info)
        if info in self.dic:
            return self.dic[info][self.type]
        else:
            assert False, info

    def predict_lat(self, spec):
        def make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            x = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # make sure that round down does not go down by more than 10%
            if x < 0.9 * v:
                x += divisor
            return x

        ans = self.other
        d = spec['d']
        ks = spec['ks']
        e = spec['e']
        w_pw, a_pw, w_dw, a_dw = spec['pw_w_bits_setting'], spec['pw_a_bits_setting'], \
                                 spec['dw_w_bits_setting'], spec['dw_a_bits_setting']
        channel_in_list = [16, 24, 40, 80, 96, 192]
        channel_out_list = [24, 40, 80, 96, 192, 320]
        input_list = []
        output_list = []
        for i in range(1, 7):
            input_list.append('{}x{}x{}'.format(self.sz_in[i], self.sz_in[i], channel_in_list[i - 1]))
            output_list.append('{}x{}x{}'.format(self.sz_out[i], self.sz_out[i], channel_out_list[i - 1]))
        # in_channels, out_channels, input_size, kernel_size, stride, padding, groups
        for layer in range(21):
            inp = input_list[layer // 4] if layer % 4 == 0 else input_list[layer // 4 + 1]
            out = output_list[layer // 4]
            input_sz = int(inp.split('x')[0])
            output_sz = int(out.split('x')[0])
            in_channels = int(inp.split('x')[-1])
            out_channels = int(out.split('x')[-1])
            stride = 1 if input_sz == output_sz else 2
            # print(inp, out)
            idskip = 1 if stride == 1 and in_channels == out_channels else 0

            mid_c = make_divisible(e[layer] * in_channels, 8)
            out_c = out_channels

            nowd = layer % 4
            stg = layer // 4
            if nowd < d[stg]:
                # in_channels, out_channels, input_size, kernel_size, stride, padding, groups
                tmp = in_channels, mid_c, input_sz, 1, 1, 0, 1  # pw
                ans += self.get_lat(tmp, (w_pw[layer], a_pw[layer]))
                tmp = mid_c, mid_c, input_sz, ks[layer], stride, ks[layer] // 2, mid_c  # dw
                assert not (ks[layer] == 7 and a_dw[layer] == 4)
                ans += self.get_lat(tmp, (w_dw[layer], a_dw[layer]))
                tmp = mid_c, out_c, input_sz // stride, 1, 1, 0, 1  # pw
                ans += self.get_lat(tmp, (w_pw[layer], a_pw[layer]))

        return ans


if __name__ == '__main__':
    lut = LatencyPredictor()
    tab = lut.build_table()
    print(tab)
    print(len(tab))
