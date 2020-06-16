# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

import yaml
from imagenet_codebase.utils import download_url


class LatencyEstimator(object):

    def __init__(self, local_dir='~/.torch/latency_tools/',
                 url='https://hanlab.mit.edu/files/proxylessNAS/LatencyTools/mobile_trim.yaml'):
        if url.startswith('http'):
            fname = download_url(url, local_dir, overwrite=True)
        else:
            fname = url

        with open(fname, 'r') as fp:
            self.lut = yaml.load(fp)

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def query(self, **kwargs):
        raise NotImplementedError

    def predict_network_latency(self, net, image_size):
        return None


class ProxylessNASLatencyEstimator(LatencyEstimator):
    def __init__(self, path=None):
        super(ProxylessNASLatencyEstimator, self).__init__()
        if path is not None:
            self.lut = yaml.load(
                open(path, 'r'))

    def query(self, l_type: str, input_shape, output_shape, expand=None, ks=None, stride=None, id_skip=None):
        """
        :param l_type:
            Layer type must be one of the followings
            1. `Conv`: The initial 3x3 conv with stride 2.
            2. `Conv_1`: feature_mix_layer
            3. `Logits`: All operations after `Conv_1`.
            4. `expanded_conv`: MobileInvertedResidual
        :param input_shape: input shape (h, w, #channels)
        :param output_shape: output shape (h, w, #channels)
        :param expand: expansion ratio
        :param ks: kernel size
        :param stride:
        :param id_skip: indicate whether has the residual connection
        """
        infos = [l_type, 'input:%s' % self.repr_shape(input_shape), 'output:%s' % self.repr_shape(output_shape), ]

        if l_type in ('expanded_conv',):
            assert None not in (expand, ks, stride, id_skip)
            infos += ['expand:%d' % expand, 'kernel:%d' % ks, 'stride:%d' % stride, 'idskip:%d' % id_skip]
        key = '-'.join(infos)
        return self.lut[key]['mean']

    def predict_network_latency(self, net, image_size=224):
        predicted_latency = 0
        # first conv
        predicted_latency += self.query(
            'Conv', [image_size, image_size, 3],
            [(image_size + 1) // 2, (image_size + 1) // 2, net.first_conv.out_channels]
        )
        # blocks
        fsize = (image_size + 1) // 2
        for block in net.blocks:
            mb_conv = block.mobile_inverted_conv
            shortcut = block.shortcut

            if mb_conv is None:
                continue
            if shortcut is None:
                idskip = 0
            else:
                idskip = 1
            out_fz = int((fsize - 1) / mb_conv.stride + 1)  # fsize // mb_conv.stride
            block_latency = self.query(
                'expanded_conv', [fsize, fsize, mb_conv.in_channels], [out_fz, out_fz, mb_conv.out_channels],
                expand=mb_conv.expand_ratio, ks=mb_conv.kernel_size, stride=mb_conv.stride, id_skip=idskip
            )
            predicted_latency += block_latency
            fsize = out_fz
        # feature mix layer
        predicted_latency += self.query(
            'Conv_1', [fsize, fsize, net.feature_mix_layer.in_channels],
            [fsize, fsize, net.feature_mix_layer.out_channels]
        )
        # classifier
        predicted_latency += self.query(
            'Logits', [fsize, fsize, net.classifier.in_features], [net.classifier.out_features]  # 1000
        )
        return predicted_latency


if __name__ == '__main__':
    latency_model = ProxylessNASLatencyEstimator()

    from imagenet_codebase.networks.proxyless_nets import MnasNet

    net_ = MnasNet()

    predicted_latency_ = latency_model.predict_network_latency(net_)
    print(predicted_latency_, 'ms')
