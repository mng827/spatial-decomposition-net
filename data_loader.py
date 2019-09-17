from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.transforms as transforms
import torchvision
import torch

import numpy as np
import nibabel as nib
import json_tricks
import os
import image_utils
import csv
from scipy import interpolate
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


def load_pytorch(config):
    mean, std = {}, {}
    mean['ukbb'] = 0.2786841
    std['ukbb'] = 0.25609297

    mean['ukbbhalf'] = 0.28025758
    std['ukbbhalf'] = 0.255864

    mean['acdc'] = 0.2688314
    std['acdc'] = 0.24191053

    mean['unknown'] = 0.0
    std['unknown'] = 1.0

    trainset_labelled = VolumetricImageDataset(data_root_dir=config.train_labelled_data_path,
                                               data=get_image_list(config.train_labelled_data_file),
                                               image_size=config.image_size,
                                               network_type=config.network_type,
                                               num_images_limit=config.num_images_limit,
                                               augment=config.data_aug,
                                               shift=config.data_aug_shift,
                                               rotate=config.data_aug_rotate,
                                               scale=config.data_aug_scale,
                                               mean=mean[config.dataset],
                                               std=std[config.dataset])

    trainset_unlabelled = VolumetricImageDataset(data_root_dir=config.train_unlabelled_data_path,
                                                 data=get_image_list(config.train_unlabelled_data_file),
                                                 image_size=config.image_size,
                                                 network_type=config.network_type,
                                                 num_images_limit=config.num_images_limit,
                                                 augment=config.data_aug,
                                                 shift=config.data_aug_shift,
                                                 rotate=config.data_aug_rotate,
                                                 scale=config.data_aug_scale,
                                                 mean=mean[config.dataset],
                                                 std=std[config.dataset])

    testset = VolumetricImageDataset(data_root_dir=config.validation_data_path,
                                     data=get_image_list(config.validation_data_file),
                                     image_size=config.image_size,
                                     network_type=config.network_type,
                                     num_images_limit=config.num_images_limit,
                                     augment=False,
                                     mean=mean[config.dataset],
                                     std=std[config.dataset])

    return torch.utils.data.DataLoader(trainset_labelled,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       num_workers=config.num_workers,
                                       collate_fn=concatenate_samples), \
           torch.utils.data.DataLoader(trainset_unlabelled,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       num_workers=config.num_workers,
                                       collate_fn=concatenate_samples), \
           torch.utils.data.DataLoader(testset,
                                       batch_size=config.test_batch_size,
                                       shuffle=False,
                                       num_workers=config.num_workers,
                                       collate_fn=concatenate_samples)


def concatenate_samples(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    info = batch[0][2]
    whole_image = [item[3] for item in batch]
    whole_label = [item[4] for item in batch]

    if any(i is None for i in label):
        return [np.concatenate(data, axis=0), None, info, whole_image, whole_label]
    else:
        return [np.concatenate(data, axis=0), np.concatenate(label, axis=0), info, whole_image, whole_label]


class VolumetricImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_root_dir, data, image_size, network_type, num_images_limit, augment, shift=10, rotate=10,
                 scale=0.1, intensity=0.0, mean=0.0, std=1.0):
        self.data_root_dir = data_root_dir
        self.data = data
        self.image_size = image_size
        self.network_type = network_type
        self.num_images_limit = num_images_limit
        self.augment = augment
        self.shift = shift
        self.rotate = rotate
        self.scale = scale
        self.intensity = intensity
        self.mean = mean
        self.std = std

    def __getitem__(self, index):
        image_name = self.data['image_filenames'][index]
        label_name = self.data['label_filenames'][index]
        phase_number = self.data['phase_numbers'][index]

        has_label = (self.data['label_filenames'][index] is not '')

        info = {}
        info['Name'] = image_name
        info['PhaseNumber'] = phase_number

        nib_image = nib.load(os.path.join(self.data_root_dir, image_name))
        info['PixelResolution'] = nib_image.header['pixdim'][1:4]
        info['ImageSize'] = nib_image.header['dim'][1:4]
        info['AffineTransform'] = nib_image.header.get_best_affine()
        info['Header'] = nib_image.header.copy()

        whole_image = nib_image.get_data()
        if has_label: whole_label = nib.load(os.path.join(self.data_root_dir, label_name)).get_data()

        if np.ndim(whole_image) == 4:
            whole_image = whole_image[:, :, :, phase_number]
            if has_label: whole_label = whole_label[:, :, :, phase_number]

        # # For ACDC
        # whole_label = 4 - whole_label
        # whole_label[whole_label == 4] = 0

        whole_image_orig = np.copy(whole_image)

        clip_min = np.percentile(whole_image, 1)
        clip_max = np.percentile(whole_image, 99)
        whole_image = np.clip(whole_image, clip_min, clip_max)
        whole_image = (whole_image - whole_image.min()) / float(whole_image.max() - whole_image.min())
        whole_image = whole_image * 2 - 1   # Make image within -1 and 1

        # whole_image = (whole_image - np.mean(whole_image, dtype=np.float32)) / np.std(whole_image, dtype=np.float32)

        # Pad image into square and resize here
        # whole_image = image_utils.zero_pad(whole_image)
        # whole_label = image_utils.zero_pad(whole_label)
        #
        # whole_image = image_utils.resize_image(whole_image, [image_size, image_size], interpolation_order=1)
        # whole_label = image_utils.resize_image(whole_label, [image_size, image_size], interpolation_order=0) * 255

        x, y, z = whole_image.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        cropped_image, _, _ = image_utils.crop_image(whole_image, x_centre, y_centre, [self.image_size, self.image_size])

        if has_label:
            cropped_label, _, _ = image_utils.crop_image(whole_label, x_centre, y_centre, [self.image_size, self.image_size])
        else:
            cropped_label = None

        # Perform data augmentation
        if self.augment:
            cropped_image, cropped_label = image_utils.augment_data_2d(cropped_image, cropped_label,
                                                                       preserve_across_slices=False,
                                                                       max_shift=self.shift, max_rotate=self.rotate,
                                                                       max_scale=self.scale, max_intensity=self.intensity)
        cropped_image = (cropped_image - self.mean) / self.std

        # !! Put into NCHW format
        batch_images = np.expand_dims(np.transpose(cropped_image, axes=(2, 0, 1)), axis=1)
        if has_label: batch_labels = np.expand_dims(np.transpose(cropped_label, axes=(2, 0, 1)), axis=1)

        if batch_images.shape[0] > self.num_images_limit:
            slices = sorted(list(np.random.permutation(batch_images.shape[0]))[:self.num_images_limit])
            batch_images = batch_images[slices, :, :, :]
            if has_label: batch_labels = batch_labels[slices, :, :, :]
            # print("Warning: Number of slices limited to {} to fit GPU".format(num_images_limit))

        if has_label:
            return batch_images, batch_labels, info, whole_image_orig, whole_label
        else:
            return batch_images, None, info, whole_image_orig, None

    def __len__(self):
        return len(self.data['image_filenames'])


def get_image_list(csv_file):
    image_list, label_list, phase_list = [], [], []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                phase_numbers = eval(row['phase_numbers'])
            except:
                print(row)
                raise Exception()

            for phase in phase_numbers:
                image_list.append(row['image_filenames'].strip())
                label_list.append(row['label_filenames'].strip())
                phase_list.append(phase)

    data_list = {}
    data_list['image_filenames'] = image_list
    data_list['label_filenames'] = label_list
    data_list['phase_numbers'] = phase_list

    return data_list
