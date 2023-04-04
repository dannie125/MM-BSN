import os
import argparse
import math
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from util.model_need_tools import set_module, set_optimizer, load_checkpoint, summary, set_status,\
    set_denoiser, test_dataloader_process, print_loss, warmup, _adjust_lr, _run_step
from util.config_parse import ConfigParser
from util.file_manager import FileManager
from util.logger import Logger
from util.loss import Loss
from util.generator import human_format
from DataDeal.Data_loader import preped_RN_data, SIDD_val, DND



def train():
    module = set_module(cfg)
    # training dataset loader
    train_args = train_cfg['dataset_args']
    train_dataset_path = os.path.join(cfg['data_root_dir'], 'prep/SIDD_s512_o128')
    train_dataset = preped_RN_data(**train_args, dataset_path=train_dataset_path)
    train_dataloader = {}
    train_dataloader['dataset'] = DataLoader(dataset=train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=cfg['thread'],
                                       pin_memory=False)

    # validation dataset loader

    val_args = val_cfg['dataset_args']
    val_dataset_path = os.path.join(cfg['data_root_dir'], 'SIDD')
    val_dataset = SIDD_val(**val_args, dataset_path=val_dataset_path)
    val_dataloader = {}
    val_dataloader['dataset'] = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                                           num_workers=cfg['thread'],
                                           pin_memory=False)
    # other configuration
    max_epoch = train_cfg['max_epoch']
    epoch = start_epoch = 1
    max_len = train_dataloader['dataset'].dataset.__len__() # base number of iteration works for dataset named 'dataset'
    max_iter = math.ceil(max_len / train_cfg['batch_size'])

    loss = Loss(train_cfg['loss'], train_cfg['tmp_info'])
    loss_dict = {'count': 0}
    tmp_info = {}
    loss_log = []

    # set optimizer
    optimizer = set_optimizer(module, train_cfg)

    for opt in optimizer.values():
        opt.zero_grad(set_to_none=True)

    # resume
    if cfg["resume"]:
        # load last checkpoint
        load_checkpoint(module, cfg, checkpoint_path=cfg['pretrained'])
        epoch = int(cfg['pretrained'].split('/')[-1].split('.')[0].split('_')[-1])+1

        # logger initialization
        logger = Logger((max_epoch, max_iter), log_dir=file_manager.get_dir(''), log_file_option='a')
    else:
        # logger initialization
        logger = Logger((max_epoch, max_iter), log_dir=file_manager.get_dir(''), log_file_option='w')

    # tensorboard
    tboard_time = datetime.datetime.now().strftime('%m-%d-%H-%M')
    tboard = SummaryWriter(log_dir=file_manager.get_dir('tboard/%s'%tboard_time))

    # device setting
    if cfg['gpu'] != 'cpu':
        # model to GPU
        model = {key: nn.DataParallel(module[key]).cuda() for key in module}
        # optimizer to GPU
        for optim in optimizer.values():
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
    else:
        model = {key: nn.DataParallel(module[key]).cpu() for key in module}

    # start message
    logger.info(summary(module, human_format))
    logger.start((epoch-1, 0))
    logger.highlight(logger.get_start_msg())

    if epoch == 1 and train_cfg['warmup']:
        warmup(model, loss, train_dataloader, optimizer, logger, max_iter, epoch, max_epoch, loss_dict, loss_log,tmp_info,tboard, cfg)

    # training
    for epoch in range(epoch, max_epoch + 1):
        status = set_status('epoch %03d/%03d' % (epoch, max_epoch))
        # make dataloader iterable.
        train_dataloader_iter = {}
        for key in train_dataloader:
            train_dataloader_iter[key] = iter(train_dataloader[key])

        # model training mode
        for key in model:
            model[key].train()

        for iter_id in range(1, max_iter+1):
            _run_step(train_dataloader_iter, model, optimizer, loss,epoch, iter_id, max_iter, max_epoch, loss_dict, cfg)
            _adjust_lr(optimizer, iter_id, epoch, max_iter, train_cfg)

            if (iter_id % cfg['log']['interval_iter'] == 0 and iter_id != 0) or (iter_id == max_iter):
                print_loss(optimizer, logger, loss_dict, loss_log, tmp_info, status, tboard, iter_id, max_iter, epoch)

            # print progress
            logger.print_prog_msg((epoch - 1, iter_id - 1))
        # save checkpoint
        ckpt_save_folder = cfg['ckpt_save_folder']
        if not os.path.exists(ckpt_save_folder):
            os.makedirs(ckpt_save_folder)
        checkpoint_name = cfg['config'].split('/')[-1] +'_'+ cfg['model']['kwargs']['type'] + '_%03d'%epoch + '.pth'
        if epoch >= ckpt_cfg['start_epoch']:
            if (epoch - ckpt_cfg['start_epoch']) % ckpt_cfg['interval_epoch'] == 0:
                torch.save({'epoch': epoch,
                            'model_weight': {key: model[key].module.state_dict() for key in model}},
                           os.path.join(ckpt_save_folder, 'checkpoint',checkpoint_name))

        # validation

        if val_cfg['val']:
            if epoch >= val_cfg['start_epoch']:
                if (epoch - val_cfg['start_epoch']) % val_cfg['interval_epoch'] == 0:
                    for key in model:
                        model[key].eval()
                    set_status('val %03d' % epoch)
                    checkpoint_path = os.path.join(ckpt_save_folder, 'checkpoint', checkpoint_name)
                    denoiser = set_denoiser(checkpoint_path, cfg)

                    # make directories for image saving
                    img_save_path = 'img/val_%03d' % epoch

                    file_manager.make_dir(img_save_path)

                    # count psnr/ssim and save denoised validation image
                    psnr, ssim = test_dataloader_process(    denoiser=denoiser,
                                                              dataloader=val_dataloader['dataset'],
                                                             file_manager=file_manager,
                                                              cfg=cfg,
                                                              add_con=0. if not 'add_con' in val_cfg else
                                                              val_cfg['add_con'],
                                                              floor=False if not 'floor' in val_cfg else
                                                              val_cfg['floor'],
                                                              img_save_path=img_save_path,
                                                              img_save=val_cfg['save_image'],
                                                             logger=logger,
                                                             status=status)

    logger.highlight(logger.get_finish_msg())

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c',  '--config',            default='config/SIDD',  type=str)
    args.add_argument('-g',  '--gpu',               default='0,1,2,3',  type=str)
    args.add_argument('-r',  '--resume',            default=False)
    args.add_argument('-p',  '--pretrained',        default=None,  type=str)
    args.add_argument('-t',  '--thread',            default=4,     type=int)
    args.add_argument('-se',  '--self_en',          action='store_true')
    args.add_argument('-sd', '--ckpt_save_folder',  default='output/MMBSN_SIDD_all', type=str)
    args.add_argument('-rd', '--data_root_dir',     default='/home/uc/proj/zhoufangfang/models/denosing/AP-BSN-master/dataset', type=str)

    args = args.parse_args()

    assert args.config is not None, 'config file path is needed'

    cfg = ConfigParser(args)

    # device setting
    if cfg['gpu'] == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    train_cfg = cfg['training']
    val_cfg = cfg['validation']
    ckpt_cfg = cfg['checkpoint']
    status_len = 13
    file_manager = FileManager(cfg['ckpt_save_folder'])

    train()