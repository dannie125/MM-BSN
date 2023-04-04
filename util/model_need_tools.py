import os
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from model.get_model import BSN
from util.generator import rot_hflip_img,ssim, psnr

@torch.no_grad()
def test_dataloader_process(denoiser, dataloader, file_manager, cfg, add_con=0., floor=False, img_save=True, img_save_path=None, info=True, logger=None, status=None):
    '''
    do test or evaluation process for each dataloader
    include following steps:
        1. denoise image
        2. calculate PSNR & SSIM
        3. (optional) save denoised image
    Args:
        dataloader : dataloader to be tested.
        add_con : add constant to denoised image.
        floor : floor denoised image. (default range is [0, 255])
        img_save : whether to save denoised and clean images.
        img_save_path (optional) : path to save denoised images.
        info (optional) : whether to print info.
    Returns:
        psnr : total PSNR score of dataloaer results or None (if clean image is not available)
        ssim : total SSIM score of dataloder results or None (if clean image is not available)
    '''
    # make directory

    file_manager.make_dir(img_save_path)

    # test start
    psnr_sum = 0.
    ssim_sum = 0.
    count = 0
    for idx, data in enumerate(dataloader):
        # to device
        if cfg['gpu'] == 'cpu':
            for key in data:
                data[key] = data[key].cpu()
        else:
            for key in data:
                data[key] = data[key].cuda()

        # forward
        input_data = [data[arg] for arg in cfg['model_input']]
        denoised_image = denoiser(*input_data)
        # print(denoised_image.shape)

        # add constant and floor (if floor is on)
        denoised_image += add_con
        if floor: denoised_image = torch.floor(denoised_image)

        # evaluation
        if 'clean' in data:
            psnr_value = psnr(denoised_image, data['clean'])
            ssim_value = ssim(denoised_image, data['clean'])

            psnr_sum += psnr_value
            ssim_sum += ssim_value
            count += 1

        # image save
        if img_save:
            # to cpu
            if 'clean' in data:
                clean_img = data['clean'].squeeze(0).cpu()
            if 'real_noisy' in cfg['model_input']:
                noisy_img = data['real_noisy']
            elif 'syn_noisy' in cfg['model_input']:
                noisy_img = data['syn_noisy']
            elif 'noisy' in cfg['model_input']:
                noisy_img = data['noisy']
            else:
                noisy_img = None
            if noisy_img is not None: noisy_img = noisy_img.squeeze(0).cpu()
            denoi_img = denoised_image.squeeze(0).cpu()

            # write psnr value on file name
            denoi_name = '%04d_DN_%.2f' % (idx, psnr_value) if 'clean' in data else '%04d_DN' % idx

            # imwrite
            if 'clean' in data:         file_manager.save_img_tensor(img_save_path, '%04d_CL' % idx, clean_img)
            if noisy_img is not None: file_manager.save_img_tensor(img_save_path, '%04d_N' % idx, noisy_img)
            file_manager.save_img_tensor(img_save_path, denoi_name, denoi_img)
            # procedure log msg
        if info:
            if 'clean' in data:
                logger.note('[%s] testing... %04d/%04d. PSNR : %.2f dB' % (
                status, idx, dataloader.__len__(), psnr_value), end='\r')
            else:
                logger.note('[%s] testing... %04d/%04d.' % (status, idx, dataloader.__len__()), end='\r')

        # final log msg
    if count > 0:
        logger.val('[%s] Done! PSNR : %.2f dB, SSIM : %.3f' % (status, psnr_sum / count, ssim_sum / count))
    else:
        logger.val('[%s] Done!' % status)
    # return
    if count == 0:
        return None, None
    else:
        return psnr_sum / count, ssim_sum / count

def set_denoiser(checkpoint_path, cfg):
    module = set_module(cfg)
    load_checkpoint(module, cfg, checkpoint_path)
    if cfg['gpu'] == 'cpu':
        model = {key: nn.DataParallel(module[key]).cpu() for key in module}
    else:
        model = {key: nn.DataParallel(module[key]).cuda() for key in module}
    if hasattr(model['denoiser'].module, 'denoise'):
        denoiser = model['denoiser'].module.denoise
    else:
        denoiser = model['denoiser'].module
    return denoiser

def set_module(cfg):
    module = {}
    if cfg['model']['kwargs'] is None:
        module['denoiser'] = BSN(cfg['model'])
    else:
        module['denoiser'] = BSN(**cfg['model']['kwargs'])
    return module

def load_checkpoint(module, cfg, checkpoint_path):

    file_name = checkpoint_path
    # check file exist
    assert os.path.isfile(file_name), 'there is no checkpoint: %s' % file_name

    if cfg['gpu'] == 'cpu':
        saved_checkpoint = torch.load(file_name, map_location=torch.device('cpu'))
    else:
        saved_checkpoint = torch.load(file_name)
    for key in module:
        module[key].load_state_dict(saved_checkpoint['model_weight'][key])

def set_optimizer(module, train_cfg):
    optimizer = {}
    for key in module:
        optimizer[key] = optim.Adam(module[key].parameters(), lr=float(train_cfg['init_lr']), betas=train_cfg['optimizer']['Adam']['betas'])
    return optimizer

def set_status(status:str, status_len=13):
    assert len(status) <= status_len, 'status string cannot exceed %d characters, (now %d)'%(status_len, len(status))

    if len(status.split(' ')) == 2:
        s0, s1 = status.split(' ')
        status = '%s'%s0.rjust(status_len//2) + ' '\
                      '%s'%s1.ljust(status_len//2)
    else:
        sp = status_len - len(status)
        status = ''.ljust(sp//2) + status + ''.ljust((sp+1)//2)
    return status

def summary(module, human_format):
    summary = ''

    summary += '-' * 100 + '\n'
    # model
    for k, v in module.items():
        # get parameter number
        param_num = sum(p.numel() for p in v.parameters())

        # get information about architecture and parameter number
        summary += '[%s] paramters: %s -->' % (k, human_format(param_num)) + '\n'
        summary += str(v) + '\n\n'

    # optim

    # Hardware

    summary += '-' * 100 + '\n'

    return summary

def _forward_fn(module, loss, data, epoch, iter_id, max_iter, max_epoch, cfg):
    # forward
    input_data = [data['dataset'][arg] for arg in cfg['model_input']]
    denoised_img = module['denoiser'](*input_data)
    model_output = {'recon': denoised_img}

    # get losses
    losses, tmp_info = loss(input_data, model_output, data['dataset'], module,
                                ratio=(epoch-1 + (iter_id-1)/max_iter)/max_epoch)

    return losses, tmp_info

def _run_step(train_dataloader_iter, model, optimizer, loss,epoch, iter_id, max_iter, max_epoch, loss_dict, cfg):
    # get data (data should be dictionary of Tensors)
    data = {}
    for key in train_dataloader_iter:
        data[key] = next(train_dataloader_iter[key])

    # to device

    for dataset_key in data:
        for key in data[dataset_key]:
            if cfg['gpu'] != 'cpu':
                data[dataset_key][key] = data[dataset_key][key].cuda()
            else:
                data[dataset_key][key] = data[dataset_key][key].cpu()

    # forward, cal losses, backward)
    losses, tmp_info = _forward_fn(model, loss, data, epoch, iter_id, max_iter, max_epoch, cfg)
    losses   = {key: losses[key].mean()   for key in losses}
    tmp_info = {key: tmp_info[key].mean() for key in tmp_info}

    # backward
    total_loss = sum(v for v in losses.values())
    total_loss.backward()

    # optimizer step
    for opt in optimizer.values():
        opt.step()

    # zero grad
    for opt in optimizer.values():
        opt.zero_grad(set_to_none=True)

    # save losses and tmp_info
    for key in losses:
        if key != 'count':
            if key in loss_dict:
                loss_dict[key] += float(losses[key])
            else:
                loss_dict[key] = float(losses[key])
    for key in tmp_info:
        if key in tmp_info:
            tmp_info[key] += float(tmp_info[key])
        else:
            tmp_info[key] = float(tmp_info[key])
    loss_dict['count'] += 1

def _adjust_lr(optimizer, iter_id, epoch, max_iter, train_cfg):
    sched = train_cfg['scheduler']

    if sched['type'] == 'step':
        '''
        step decreasing scheduler
        Args:
            step_size: step size(epoch) to decay the learning rate
            gamma: decay rate
        '''
        if iter_id == max_iter:
            args = sched['step']
            if epoch % args['step_size'] == 0:
                for optimizer in optimizer.values():
                    lr_before = optimizer.param_groups[0]['lr']
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_before * float(args['gamma'])
                        return param_group["lr"]
    elif sched['type'] == 'linear':
        '''
        linear decreasing scheduler
        Args:
            step_size: step size(epoch) to decrease the learning rate
            gamma: decay rate for reset learning rate
        '''
        args = sched['linear']
        reset_lr = float(train_cfg['init_lr']) * float(args['gamma'])**((epoch-1)//args['step_size'])

        # reset lr to initial value
        if epoch % args['step_size'] == 0 and iter_id == max_iter:
            reset_lr = float(train_cfg['init_lr']) * float(args['gamma'])**(epoch//args['step_size'])
            for optimizer in optimizer.values():
                for param_group in optimizer.param_groups:
                    param_group["lr"] = reset_lr
                    return param_group["lr"]
        # linear decaying
        else:
            ratio = ((epoch + (iter_id)/max_iter - 1) % args['step_size']) / args['step_size']
            curr_lr = (1-ratio) * reset_lr
            for optimizer in optimizer.values():
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr
                    return param_group["lr"]

    else:
        raise RuntimeError('ambiguious scheduler type: {}'.format(sched['type']))

def _get_current_lr(optimizer):
    for first_optim in optimizer.values():
        for param_group in first_optim.param_groups:
            return param_group['lr']

def print_loss(optimizer, logger, loss_dict, loss_log,tmp_info, status, tboard, iter_id, max_iter,epoch):
    temporal_loss = 0.
    for key in loss_dict:
        if key != 'count':
                temporal_loss += loss_dict[key]/loss_dict['count']
    loss_log += [temporal_loss]
    if len(loss_log) > 100: loss_log.pop(0)

    # print status and learning rate
    # print('........................', _get_current_lr(optimizer))
    loss_out_str = '[%s] %04d/%04d, lr:%s ∣ '%(status, iter_id, max_iter, "{:.1e}".format(_get_current_lr(optimizer)))
    global_iter = (epoch-1)*max_iter + iter_id

    # print losses
    avg_loss = np.mean(loss_log)
    loss_out_str += 'avg_100 : %.3f ∣ '%(avg_loss)
    tboard.add_scalar('loss/avg_100', avg_loss, global_iter)

    for key in loss_dict:
        if key != 'count':
            loss = loss_dict[key]/loss_dict['count']
            loss_out_str += '%s : %.3f ∣ '%(key, loss)
            tboard.add_scalar('loss/%s'%key, loss, global_iter)
            loss_dict[key] = 0.

    # print temporal information
    if len(tmp_info) > 0:
        loss_out_str += '\t['
        for key in tmp_info:
            loss_out_str += '  %s : %.2f'%(key, tmp_info[key]/loss_dict['count'])
            tmp_info[key] = 0.
        loss_out_str += ' ]'

    # reset
    loss_dict['count'] = 0
    logger.info(loss_out_str)

def warmup(model, loss, train_dataloader, optimizer, logger, max_iter, epoch, max_epoch, loss_dict, loss_log,tmp_info,tboard, cfg):
    train_cfg = cfg['training']

    status = set_status('warmup')
    # make dataloader iterable.
    train_dataloader_iter = {}
    for key in train_dataloader:
        train_dataloader_iter[key] = iter(train_dataloader[key])

    warmup_iter = train_cfg['warmup_iter']
    if warmup_iter > max_iter:
        logger.info('currently warmup support 1 epoch as maximum. warmup iter is replaced to 1 epoch iteration. %d -> %d' \
            % (warmup_iter, max_iter))
        warmup_iter = max_iter

    for iter_id in range(1, warmup_iter+1):
        init_lr = float(train_cfg['init_lr'])
        warmup_lr = init_lr * iter_id / warmup_iter

        for optimizer in optimizer.values():
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr


        _run_step(train_dataloader_iter, model, optimizer, loss,epoch, iter_id, max_iter, max_epoch, loss_dict, cfg)
        _adjust_lr(optimizer, iter_id, epoch, max_iter, train_cfg)
        if (iter_id % cfg['log']['interval_iter'] == 0 and iter_id != 0) or (iter_id == max_iter):
            print_loss(optimizer, logger, loss_dict, loss_log,tmp_info, status, tboard, iter_id, max_iter,epoch)

        # print progress
        logger.print_prog_msg((epoch - 1, iter_id - 1))

@torch.no_grad()
def self_ensemble(fn, x):
    '''
    Geomery self-ensemble function
    Note that in this function there is no gradient calculation.
    Args:
        fn : denoiser function
        x : input image
    Return:
        result : self-ensembled image
    '''
    result = torch.zeros_like(x)

    for i in range(8):
        tmp = fn(rot_hflip_img(x, rot_times=i%4, hflip=i//4))
        tmp = rot_hflip_img(tmp, rot_times=4-i%4)
        result += rot_hflip_img(tmp, hflip=i//4)
    return result / 8

@torch.no_grad()
def crop_test(fn, x, size=512, overlap=0):
    '''
    crop test image and inference due to memory problem
    '''
    b, c, h, w = x.shape
    denoised = torch.zeros_like(x)
    for i in range(0, h, size - overlap):
        for j in range(0, w, size - overlap):
            end_i = min(i + size, h)
            end_j = min(j + size, w)
            x_crop = x[..., i:end_i, j:end_j]
            denoised_crop = fn(x_crop)

            start_i = overlap if i != 0 else 0
            start_j = overlap if j != 0 else 0

            denoised[..., i + start_i:end_i, j + start_j:end_j] = denoised_crop[..., start_i:, start_j:]

    return denoised