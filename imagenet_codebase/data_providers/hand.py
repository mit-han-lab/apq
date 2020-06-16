# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

import warnings
import os
import math

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from imagenet_codebase.data_providers.base_provider import DataProvider
import pickle
import PIL.Image


class Hand(torch.utils.data.Dataset):
    """Hand dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _target_transform, callable: A function/transform that takes in the
            target and transforms it.
        _train_data, list of np.ndarray.
        _train_labels, list of int.
        _test_data, list of np.ndarray.
        _test_labels, list of int.
    """

    def __init__(self, root='~/dataset/hand', train=True, transform=None, target_transform=None,
                 download=False):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
            transform, callable [None]: A function/transform that takes in a
                PIL.Image and transforms it.
            target_transform, callable [None]: A function/transform that takes
                in the target and transforms it.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        os.makedirs(self._root, exist_ok=True)
        self._train = train
        self._transform = transform
        self._target_transform = target_transform

        # Now load the picked data.
        if self._train:
            self._tmp_data, self._tmp_labels = pickle.load(open(
                os.path.join(self._root, 'train.pkl'), 'rb'))
            import numpy as np
            rep = 3
            self._train_data = self._tmp_data * rep
            self._train_labels = self._tmp_labels * rep
            assert (len(self._train_data) == 74 * rep
                    and len(self._train_labels) == 74 * rep)
        else:
            self._test_data, self._test_labels = pickle.load(open(
                os.path.join(self._root, 'test.pkl'), 'rb'))
            assert (len(self._test_data) == 38
                    and len(self._test_labels) == 38)

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.

        Returns:
            image, PIL.Image: Image of the given index.
            target, str: target of the given index.
        """
        if self._train:
            image, target = self._train_data[index], self._train_labels[index]
        else:
            image, target = self._test_data[index], self._test_labels[index]
        # Doing this so that it is consistent with all other datasets.
        image = PIL.Image.fromarray(image)
        # print(np.array(image))

        if self._transform is not None:
            image = self._transform(image)
        if self._target_transform is not None:
            target = self._target_transform(target)
        # print(image)

        return image, target

    @property
    def samples(self):
        return self._train_data

    def __len__(self):
        """Length of the dataset.

        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_data)
        return len(self._test_data)


class HandDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=128, test_batch_size=128, valid_size=None, n_worker=10,
                 resize_scale=0.08, distort_color=None, image_size=320):

        warnings.filterwarnings('ignore')
        self._save_path = save_path

        self.image_size = image_size  # int or list of int
        self.distort_color = distort_color
        self.resize_scale = resize_scale

        self.active_img_size = self.image_size

        valid_transforms = transforms.Compose([
            transforms.Resize(int(math.ceil(self.active_img_size))),
            # transforms.CenterCrop(self.active_img_size),
            transforms.ToTensor(),
            self.normalize,
        ])
        train_transforms = self.build_train_transform()
        self._valid_transform_dict = {self.active_img_size: valid_transforms}
        self.transform_ = valid_transforms
        train_dataset = Hand(self.save_path, train=True, transform=train_transforms)

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size

            valid_dataset = Hand(self.save_path, train=True, transform=valid_transforms)
            train_indexes, valid_indexes = self.random_sample_valid_set(
                [cls for _, cls in train_dataset.samples], valid_size, self.n_classes,
            )
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = None

        test_dataset = Hand(self.save_path, train=False, transform=valid_transforms)
        self.test = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
        )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'hand'

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 1

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/dataset/hand'
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser('~/dataset/hand')
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download Hand')

    def train_dataset(self, _transforms):
        return Hand(self.save_path, train=True, transform=_transforms)

    def test_dataset(self, _transforms):
        return datasets.ImageFolder(self.valid_path, _transforms)

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self.save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = transforms.Compose([
                transforms.Resize(int(math.ceil(self.active_img_size))),
                # transforms.CenterCrop(self.active_img_size),
                transforms.ToTensor(),
                self.normalize,
            ])
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, resize_scale: %s, img_size: %s' %
                  (self.distort_color, self.resize_scale, image_size))
        if self.distort_color == 'torch':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif self.distort_color == 'tf':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None

        if color_transform is None:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(self.resize_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(self.resize_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                color_transform,
                transforms.ToTensor(),
                self.normalize,
            ])
        return train_transforms

    def build_sub_train_loader(self, n_images, batch_size, num_worker=None):
        if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset.samples)
            g = torch.Generator()
            g.manual_seed(DataProvider.VALID_SEED)  # set random seed before sampling validation set
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = Hand(self.save_path, train=True, transform=self.build_train_transform(print_log=False))
            chosen_indexes = rand_indexes[:n_images]
            sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
                num_workers=num_worker, pin_memory=True,
            )
            self.__dict__['sub_train_%d' % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__['sub_train_%d' % self.active_img_size].append((images, labels))
        return self.__dict__['sub_train_%d' % self.active_img_size]
