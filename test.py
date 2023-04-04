import os
import argparse
import cv2
import numpy as np

import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader

from util.generator import np2tensor, tensor2np
from util.config_parse import ConfigParser
from util.file_manager import FileManager
from util.logger import Logger
from util.model_need_tools import set_denoiser,test_dataloader_process, crop_test, self_ensemble, set_status

from DataDeal.dnd_submission.bundle_submissions import bundle_submissions_srgb
from DataDeal.dnd_submission.dnd_denoise import denoise_srgb
from DataDeal.dnd_submission.pytorch_wrapper import pytorch_denoiser
from DataDeal.Data_loader import SIDD, SIDD_benchmark, SIDD_val, DND, preped_RN_data




def test_img(denoiser, image_path, save_dir='./'):
    '''
    Inference a single image.
    '''
    # load image
    print(image_path)
    noisy = np2tensor(cv2.imread(image_path))
    noisy = noisy.unsqueeze(0).float()

    # to device
    if cfg['gpu'] == 'cpu':
        noisy = noisy.cpu()
    else:
        noisy = noisy.cuda()

    denoised = denoiser(noisy)
    # post-process
    denoised += test_cfg['add_con']
    if test_cfg['floor']: denoised = torch.floor(denoised)

    # save image
    denoised = tensor2np(denoised)
    denoised = denoised.squeeze(0)

    name = image_path.split('/')[-1].split('.')[0]
    cv2.imwrite(os.path.join(save_dir, name + '_DN.png'), denoised)
    # print message
    print('saved : %s' % (os.path.join(save_dir, name + '_DN.png')))

def test_dir(denoiser, direc, save_dir):
    '''
    Inference all images in the directory.
    '''
    for f in os.listdir(direc):
        test_img(denoiser, os.path.join(direc, f), save_dir)

def test_DND(img_save_path, file_manager):
    '''
    Benchmarking DND dataset.
    '''
    # make directories for .mat & image saving
    file_manager.make_dir(img_save_path)
    file_manager.make_dir(img_save_path + '/mat')
    if test_cfg['save_image']: file_manager.make_dir(img_save_path + '/img')

    def wrap_denoiser(Inoisy, nlf, idx, kidx):
        noisy = 255 * torch.from_numpy(Inoisy)

        # to device
        if cfg['gpu'] == 'cpu':
            noisy = noisy.cpu()
        else:
            noisy = noisy.cuda()

        noisy = autograd.Variable(noisy)

        # processing
        noisy = noisy.permute(2,0,1)

        noisy = noisy.view(1,noisy.shape[0], noisy.shape[1], noisy.shape[2])

        denoised = denoiser(noisy)

        denoised += test_cfg['add_con']
        if test_cfg['floor']: denoised = torch.floor(denoised)

        denoised = denoised[0,...].cpu().numpy()
        denoised = np.transpose(denoised, [1,2,0])

        # image save
        if test_cfg['save_image'] and False:
            file_manager.save_img_numpy(img_save_path+'/img', '%02d_%02d_N'%(idx, kidx),  255*Inoisy)
            file_manager.save_img_numpy(img_save_path+'/img', '%02d_%02d_DN'%(idx, kidx), denoised)

        return denoised / 255
    DND_infor_path = cfg['data_root_dir'] + '/DND/dnd_2017'
    denoise_srgb(wrap_denoiser, DND_infor_path, file_manager.get_dir(img_save_path+'/mat'))

    bundle_submissions_srgb(file_manager.get_dir(img_save_path+'/mat'))

@torch.no_grad()
def test():
    ''' initialization test setting '''
    # initialization
    # forward
    img_save_path = 'img/test_%s' % (cfg['test']['dataset'])
    # -- [ TEST Images ] -- #
    if cfg['test_dir'] is not None:
        test_dir(denoiser, cfg['test_dir'], cfg['save_folder'])

    # -- [ TEST DND Benchmark ] -- #
    elif test_cfg['dataset'] == 'DND_benchmark':
        file_manager = FileManager(output_folder)

        test_DND(img_save_path, file_manager)
    # -- [ Test Normal Dataset ] -- #
    else:
        file_manager = FileManager(output_folder)

        if test_cfg['dataset'] == 'SIDD_benchmark':
            test_args = test_cfg['dataset_args']
            dataset_path = os.path.join(cfg['data_root_dir'], 'SIDD')
            test_dataset = SIDD_benchmark(**test_args, dataset_path=dataset_path)

        elif test_cfg['dataset'] == 'SIDD_val':
            test_args = test_cfg['dataset_args']
            dataset_path = os.path.join(cfg['data_root_dir'], 'SIDD')
            test_dataset = SIDD_val(**test_args, dataset_path=dataset_path)
        else:
            test_args = test_cfg['dataset_args']
            dataset_path = os.path.join(cfg['data_root_dir'], test_cfg['dataset'])
            test_dataset = preped_RN_data(**test_args, dataset_path=dataset_path)
        dataloader = {}
        dataloader['dataset'] = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=cfg['thread'],
                                           pin_memory=False)
        test_dataloader = dataloader

        psnr, ssim = test_dataloader_process(   denoiser=denoiser,
                                                file_manager=file_manager,
                                                cfg=cfg,
                                                dataloader    = test_dataloader['dataset'],
                                                add_con       = 0.  if not 'add_con' in test_cfg else test_cfg['add_con'],
                                                floor         = False if not 'floor' in test_cfg else test_cfg['floor'],
                                                img_save_path = img_save_path,
                                                img_save      = test_cfg['save_image'],
                                                logger=logger,
                                                status=status)

        # print out result as filename
        if psnr is not None and ssim is not None:
            with open(os.path.join(file_manager.get_dir(img_save_path), '_psnr-%.2f_ssim-%.3f.result'%(psnr, ssim)), 'w') as f:
                f.write('PSNR: %f\nSSIM: %f'%(psnr, ssim))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default='config/SIDD',  type=str)
    args.add_argument('-e', '--ckpt_epoch',   default=0,     type=int)
    args.add_argument('-g', '--gpu',          default='0',  type=str)
    args.add_argument(      '--save_folder', default='output/SIDD_test_out', type=str)
    args.add_argument(      '--pretrained',   default='./ckpt/MMBSN_SIDD_o_a45.pth',  type=str)
    args.add_argument(      '--thread',       default=4,     type=int)
    args.add_argument(      '--self_en',      action='store_true')
    args.add_argument(      '--test_dir',     default='./dataset/test_data',  type=str)
    args.add_argument('-rd', '--data_root_dir',
                      default='./dataset', type=str)

    args = args.parse_args()

    assert args.config is not None, 'config file path is needed'

    cfg = ConfigParser(args)

    # device setting
    if cfg['gpu'] == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    test_cfg = cfg['test']
    ckpt_cfg = cfg['checkpoint']
    status_len = 13
    output_folder = cfg['save_folder']
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    print(cfg['pretrained'])
    logger = Logger()
    logger.highlight(logger.get_start_msg())
    status = set_status('test')
    denoiser = set_denoiser(checkpoint_path=cfg['pretrained'], cfg=cfg)
    # status = set_status('test%03d'%cfg['ckpt_epoch'])
    if cfg['self_en']:
        denoiser = lambda *input_data: self_ensemble(denoiser, *input_data)
    elif 'crop' in cfg['test']:
        denoiser = lambda *input_data: crop_test(denoiser, *input_data, size=cfg['test']['crop'])

    test()