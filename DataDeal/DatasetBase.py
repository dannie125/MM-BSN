import numpy as np
import random, os
import cv2

import torch
from torch.utils.data import Dataset, DataLoader


from util.generator import rot_hflip_img, tensor2np, np2tensor, mean_conv2d


def set_dataloader(dataset_class, dataset_cfg, batch_size, shuffle, num_workers):
    dataloader = {}
    dataset_dict = dataset_cfg['dataset']
    if not isinstance(dataset_dict, dict):
        dataset_dict = {'dataset': dataset_dict}

    for key in dataset_dict:
        args = dataset_cfg[key + '_args']
        dataset = dataset_class(dataset_dict[key])(**args)
        dataloader[key] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                     pin_memory=False)
    return dataloader


def is_image_tensor(x):
    '''
    return input tensor has image shape. (include batched image)
    '''
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3 or len(x.shape) == 4:
            if x.dtype != torch.bool:
                return True
    return False


def get_patch(crop_size, data, rnd=True):
    # check all image size is same
    if 'clean' in data and 'real_noisy' in data:
        assert data['clean'].shape[1] == data['clean'].shape[1] and data['real_noisy'].shape[2] == \
               data['real_noisy'].shape[2], \
            'img shape should be same. (%d, %d) != (%d, %d)' % (
            data['clean'].shape[1], data['clean'].shape[1], data['real_noisy'].shape[2], data['real_noisy'].shape[2])

    # get image shape and select random crop location
    if 'clean' in data:
        max_x = data['clean'].shape[2] - crop_size[0]
        max_y = data['clean'].shape[1] - crop_size[1]
    else:
        max_x = data['real_noisy'].shape[2] - crop_size[0]
        max_y = data['real_noisy'].shape[1] - crop_size[1]

    assert max_x >= 0
    assert max_y >= 0

    if rnd and max_x > 0 and max_y > 0:
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
    else:
        x, y = 0, 0

    # crop
    if 'clean' in data:
        data['clean'] = data['clean'][:, y:y + crop_size[1], x:x + crop_size[0]]
    if 'real_noisy' in data:
        data['real_noisy'] = data['real_noisy'][:, y:y + crop_size[1], x:x + crop_size[0]]

    return data


def augmentation(data: dict, aug: list):
    '''
    Parsing augmentation list and apply it to the data images.
    '''
    # parsign augmentation
    rot, hflip = 0, 0
    for aug_name in aug:
        # aug : random rotation
        if aug_name == 'rot':
            rot = random.randint(0, 3)
        # aug : random flip
        elif aug_name == 'hflip':
            hflip = random.randint(0, 1)
        else:
            raise RuntimeError('undefined augmentation option : %s' % aug_name)

    # for every data(only image), apply rotation and flipping augmentation.
    for key in data:
        if is_image_tensor(data[key]):
            # random rotation and flip
            if rot != 0 or hflip != 0:
                data[key] = rot_hflip_img(data[key], rot, hflip)
    return data

def save_all_image(save_dir, img_paths, data, clean=False, syn_noisy=False, real_noisy=False):
    for idx in range(len(img_paths)):
        if clean and 'clean' in data:
            cv2.imwrite(os.path.join(save_dir, '%04d_CL.png' % idx), tensor2np(data['clean']))
        if syn_noisy and 'syn_noisy' in data:
            cv2.imwrite(os.path.join(save_dir, '%04d_SN.png' % idx), tensor2np(data['syn_noisy']))
        if real_noisy and 'real_noisy' in data:
            cv2.imwrite(os.path.join(save_dir, '%04d_RN.png' % idx), tensor2np(data['real_noisy']))
        print('image %04d saved!' % idx)


class RealDataSet(Dataset):
    def __init__(self, crop_size: list = None, aug: list = None, n_repeat: int = 1,
                 n_data: int = None, ratio_data: float = None, dataset_path: str = None) -> None:
        '''
        Base denoising dataset class for various dataset.

        to build custom dataset class, below functions must be implemented in the inherited class. (or see other dataset class already implemented.)
            - self._scan(self) : scan image data & save its paths. (saved to self.img_paths)
            - self._load_data(self, data_idx) : load single paired data from idx as a form of dictionary.

        Args:
            add_noise (str)     : configuration of additive noise to synthesize noisy image. (see _add_noise() for more details.)
            crop_size (list)    : crop size, e.g. [W] or [H, W] and no crop if None
            aug (list)          : list of data augmentations (see _augmentation() for more details.)
            n_repeat (int)      : number of repeat for each data.
            n_data (int)        : number of data to be used. (default: None = all data)
            ratio_data (float)  : ratio of data to be used. (activated when n_data=None, default: None = all data)
        '''
        self.dataset_path = dataset_path
        if not os.path.isdir(self.dataset_path):
            raise Exception('dataset directory is not exist')
        # set parameters for dataset.
        self.crop_size = crop_size
        self.aug = aug
        self.n_repeat = n_repeat

        # scan all data and fill in self.img_paths
        self.img_paths = []
        self._scan()
        if len(self.img_paths) > 0:
            if self.img_paths[0].__class__.__name__ in ['int', 'str', 'float']:
                self.img_paths.sort()

        # set data amount
        if n_data is not None:
            self.n_data = n_data
        elif ratio_data is not None:
            self.n_data = int(ratio_data * len(self.img_paths))
        else:
            self.n_data = len(self.img_paths)

    def __len__(self):
        return self.n_data * self.n_repeat

    def __getitem__(self, idx):
        '''
        final dictionary shape of data:
        {'clean', 'syn_noisy', 'real_noisy', 'noisy (any of real[first priority] and syn)', etc}
        '''
        # calculate data index
        data_idx = idx % self.n_data

        # load data
        data = self._load_data(data_idx)

        # pre-processing (currently only crop)
        if self.crop_size != None:
            data = get_patch(self.crop_size, data)

        # data augmentation
        if self.aug is not None:
            data = augmentation(data, self.aug)
        # add general label 'noisy' to use any of real_noisy or syn_noisy (real first)
        if 'real_noisy' in data or 'syn_noisy' in data:
            data['noisy'] = data['real_noisy'] if 'real_noisy' in data else data['syn_noisy']
        return data

    def _scan(self):
        raise NotImplementedError
        # TODO fill in self.img_paths (include path from project directory)

    def _load_data(self, data_idx):
        raise NotImplementedError
        # TODO load possible data as dictionary
        # dictionary key list :
        #   'clean' : clean image without noise (gt or anything).
        #   'real_noisy' : real noisy image or already synthesized noisy image.
        #   'instances' : any other information of capturing situation.

    # ----------------------------#
    #  Image handling functions  #
    # ----------------------------#
    def _load_img(self, img_name, as_gray=False):
        img = cv2.imread(img_name, 1)
        assert img is not None, "failure on loading image - %s" % img_name
        return self._load_img_from_np(img, as_gray, RGBflip=True)

    def _load_img_from_np(self, img, as_gray=False, RGBflip=False):
        # if color
        if len(img.shape) != 2:
            if as_gray:
                # follows definition of sRBG in terms of the CIE 1931 linear luminance.
                # because calculation opencv color conversion and imread grayscale mode is a bit different.
                # https://en.wikipedia.org/wiki/Grayscale
                img = np.average(img, axis=2, weights=[0.0722, 0.7152, 0.2126])
                img = np.expand_dims(img, axis=0)
            else:
                if RGBflip:
                    img = np.flip(img, axis=2)
                img = np.transpose(img, (2, 0, 1))
        # if gray
        else:
            img = np.expand_dims(img, axis=0)
        return torch.from_numpy(np.ascontiguousarray(img).astype(np.float32))


