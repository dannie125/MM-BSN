import os
import scipy.io
import numpy as np
import h5py

import torch
from .DatasetBase import RealDataSet

class SIDD(RealDataSet):
    '''
    SIDD datatset class using original images.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        dataset_path = self.dataset_path
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        # scan all image path & info in dataset path
        for folder_name in os.listdir(dataset_path):
            # parse folder name of each shot
            parsed_name = self._parse_folder_name(folder_name)

            # add path & information of image 0
            info0 = {}
            info0['instances'] = parsed_name
            info0['clean_img_path'] = os.path.join(dataset_path, folder_name, '%s_GT_SRGB_010.PNG'%parsed_name['scene_instance_number'])
            info0['noisy_img_path'] = os.path.join(dataset_path, folder_name, '%s_NOISY_SRGB_010.PNG'%parsed_name['scene_instance_number'])
            self.img_paths.append(info0)

            # add path & information of image 1
            info1 = {}
            info1['instances'] = parsed_name
            info1['clean_img_path'] = os.path.join(dataset_path, folder_name, '%s_GT_SRGB_011.PNG'%parsed_name['scene_instance_number'])
            info1['noisy_img_path'] = os.path.join(dataset_path, folder_name, '%s_NOISY_SRGB_011.PNG'%parsed_name['scene_instance_number'])
            self.img_paths.append(info1)

    def _load_data(self, data_idx):
        info = self.img_paths[data_idx]

        clean_img = self._load_img(info['clean_img_path'])
        noisy_img = self._load_img(info['noisy_img_path'])

        return {'clean': clean_img, 'real_noisy': noisy_img, 'instances': info['instances']}

    def _parse_folder_name(self, name):
        parsed = {}
        splited = name.split('_')
        parsed['scene_instance_number']      = splited[0]
        parsed['scene_number']               = splited[1]
        parsed['smartphone_camera_code']     = splited[2]
        parsed['ISO_speed']                  = splited[3]
        parsed['shutter_speed']              = splited[4]
        parsed['illuminant_temperature']     = splited[5]
        parsed['illuminant_brightness_code'] = splited[6]
        return parsed


class SIDD_benchmark(RealDataSet):
    '''
    SIDD benchmark dataset class
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        dataset_path = self.dataset_path
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        mat_file_path = os.path.join(dataset_path, 'BenchmarkNoisyBlocksSrgb.mat')

        self.noisy_patches = np.array(scipy.io.loadmat(mat_file_path, appendmat=False)['BenchmarkNoisyBlocksSrgb'])

        # for __len__(), make img_paths have same length
        # number of all possible patch is 1280
        for _ in range(1280):
            self.img_paths.append(None)

    def _load_data(self, data_idx):
        img_id   = data_idx // 32
        patch_id = data_idx  % 32

        noisy_img = self.noisy_patches[img_id, patch_id, :].astype(float)

        noisy_img = self._load_img_from_np(noisy_img)

        return {'real_noisy': noisy_img}

class SIDD_val(RealDataSet):
    '''
    SIDD validation dataset class
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        dataset_path = self.dataset_path
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        clean_mat_file_path = os.path.join(dataset_path, 'ValidationGtBlocksSrgb.mat')
        noisy_mat_file_path = os.path.join(dataset_path, 'ValidationNoisyBlocksSrgb.mat')

        self.clean_patches = np.array(scipy.io.loadmat(clean_mat_file_path, appendmat=False)['ValidationGtBlocksSrgb'])
        self.noisy_patches = np.array(scipy.io.loadmat(noisy_mat_file_path, appendmat=False)['ValidationNoisyBlocksSrgb'])

        # for __len__(), make img_paths have same length
        # number of all possible patch is 1280
        for _ in range(1280):
            self.img_paths.append(None)

    def _load_data(self, data_idx):
        img_id   = data_idx // 32
        patch_id = data_idx  % 32

        clean_img = self.clean_patches[img_id, patch_id, :].astype(float)
        noisy_img = self.noisy_patches[img_id, patch_id, :].astype(float)

        clean_img = self._load_img_from_np(clean_img)
        noisy_img = self._load_img_from_np(noisy_img)

        return {'clean': clean_img, 'real_noisy': noisy_img }

class DND(RealDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        dataset_path = self.dataset_path
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path
        for root, _, files in os.walk(dataset_path):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        with h5py.File(self.img_paths[data_idx], 'r') as img_file:
            noisy_img = img_file[list(img_file.keys())[0]][()]*255.
        return {'real_noisy': torch.from_numpy(noisy_img)}

class clean_add_syn(RealDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        dataset_path = self.dataset_path
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path
        for root, _, files in os.walk(dataset_path):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        with h5py.File(self.img_paths[data_idx], 'r') as img_file:
            noisy_img = img_file[list(img_file.keys())[0]][()]*255.
        return {'real_noisy': torch.from_numpy(noisy_img)}


class preped_RN_data(RealDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        self.dataset_path = self.dataset_path
        assert os.path.exists(self.dataset_path), 'There is no dataset %s'%self.dataset_path
        for root, _, files in os.walk(os.path.join(self.dataset_path, 'RN')):
            self.img_paths = files

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]

        noisy_img = self._load_img(os.path.join(self.dataset_path, 'RN', file_name))

        return {'real_noisy': noisy_img} #'instances': instance }


class Rain(RealDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        # check if the dataset exists
        dataset_path = self.dataset_path
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        # WRITE YOUR CODE FOR SCANNING DATA
        # example:
        for root, _, files in os.walk(os.path.join(self.dataset_path, 'RN')):
                self.img_paths= files

    def _load_data(self, data_idx):
        # WRITE YOUR CODE FOR LOADING DATA FROM DATA INDEX
        # example:
        file_name = self.img_paths[data_idx]
        print(file_name)

        noisy_img = self._load_img(os.path.join(self.dataset_path, 'RN', file_name))
        clean_img = self._load_img(os.path.join(self.dataset_path, 'CL', file_name))
        # clean_img = self._load_img(file_name)
        # print(clean_img.shape)
        return {'clean': clean_img, 'real_noisy': noisy_img} # paired dataset
        # return {'clean': clean_img} # only noisy image dataset






