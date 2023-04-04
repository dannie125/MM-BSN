import os
import argparse
import cv2

from util.generator import tensor2np
from DataDeal.Data_loader import SIDD, SIDD_benchmark, DND, Rain

def prep_save(img_idx: int, img_size: int, overlap: int, clean: bool = False, syn_noisy: bool = False,
              real_noisy: bool = False, dataset_dir: str = None):
    '''
    cropping am image into mini-size patches for efficient training.
    Args:
        img_idx (int) : index of image
        img_size (int) : size of image
        overlap (int) : overlap between patches
        clean (bool) : save clean image (default: False)
        syn_noisy (bool) : save synthesized noisy image (default: False)
        real_noisy (bool) : save real noisy image (default: False)
    '''
    d_name = '%s_s%d_o%d' % (args.dataset_name, img_size, overlap)
    os.makedirs(os.path.join(dataset_dir, 'prep', d_name), exist_ok=True)

    assert overlap < img_size
    stride = img_size - overlap

    if clean:
        clean_dir = os.path.join(dataset_dir, 'prep', d_name, 'CL')
        os.makedirs(clean_dir, exist_ok=True)
    if syn_noisy:
        syn_noisy_dir = os.path.join(dataset_dir, 'prep', d_name, 'SN')
        os.makedirs(syn_noisy_dir, exist_ok=True)
    if real_noisy:
        real_noisy_dir = os.path.join(dataset_dir, 'prep', d_name, 'RN')
        os.makedirs(real_noisy_dir, exist_ok=True)
    data = dataset_type.__getitem__(img_idx)
    c, h, w = data['clean'].shape if 'clean' in data else data['real_noisy'].shape
    for h_idx in range((h - img_size) // stride + 1):
        for w_idx in range((w - img_size + 1) // stride + 1):
            hl, hr = h_idx * stride, h_idx * stride + img_size
            wl, wr = w_idx * stride, w_idx * stride + img_size
            if clean:      cv2.imwrite(os.path.join(clean_dir, '%d_%d_%d.png' % (img_idx, h_idx, w_idx)),
                                       tensor2np(data['clean'][:, hl:hr, wl:wr]))
            if syn_noisy:  cv2.imwrite(os.path.join(syn_noisy_dir, '%d_%d_%d.png' % (img_idx, h_idx, w_idx)),
                                       tensor2np(data['syn_noisy'][:, hl:hr, wl:wr]))
            if real_noisy: cv2.imwrite(os.path.join(real_noisy_dir, '%d_%d_%d.png' % (img_idx, h_idx, w_idx)),
                                       tensor2np(data['real_noisy'][:, hl:hr, wl:wr]))

    print('Cropped image %d/%d' % (img_idx, dataset_type.__len__()))


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('-d', '--dataset_name',    default='Rain',   type=str)
    args.add_argument('-rd', '--data_root_dir', default=r'D:\work_zff\datasets\yu', type=str)
    args.add_argument('-s', '--patch_size', default=512,  type=int)
    args.add_argument('-o', '--overlap',    default=128,  type=int)
    args.add_argument('-p', '--process',    default=8,    type=int)

    args = args.parse_args()
    # check what the dataset have images
    if args.dataset_name == 'SIDD':
        dataset_path = os.path.join(args.data_root_dir, 'SIDD/SIDD_Medium_Srgb/Data')
        dataset_type = SIDD(dataset_path=dataset_path)
    elif args.dataset_name == 'SIDD_benchmark':
        dataset_path = os.path.join(args.data_root_dir, 'SIDD')
        dataset_type = SIDD_benchmark(dataset_path=dataset_path)
    elif args.dataset_name == 'DND':
        dataset_path = os.path.join(args.data_root_dir, 'DND/dnd_2017/images_srgb')
        dataset_type = DND(dataset_path=dataset_path)
    elif args.dataset_name == 'Rain':
        dataset_path = os.path.join(args.data_root_dir, 'val')
        dataset_type = Rain(dataset_path=dataset_path)
    else:
        print('defult dataset name is SIDD')
        dataset_path = os.path.join(args.data_root_dir, 'SIDD/SIDD_Medium_Srgb/Data')
        dataset_type = SIDD(dataset_dir=dataset_path)

    data_sample = dataset_type.__getitem__(0)
    flag_c, flag_n = 'clean' in data_sample, 'real_noisy' in data_sample

    for data_idx in range(dataset_type.__len__()):
        prep_save(data_idx, args.patch_size, args.overlap, flag_c, False, flag_n, dataset_dir=args.data_root_dir)

