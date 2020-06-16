# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

from imagenet_codebase.run_manager.run_manager import *
from imagenet_codebase.data_providers.imagenet import *
from imagenet_codebase.data_providers.svhn import *
from imagenet_codebase.data_providers.coil import *
from imagenet_codebase.data_providers.hand import *


class ImagenetRunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='imagenet', train_batch_size=256, test_batch_size=512, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='tf', image_size=224, **kwargs):
        super(ImagenetRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            elif self.dataset == ImageNet100DataProvider.name():
                DataProviderClass = ImageNet100DataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']
