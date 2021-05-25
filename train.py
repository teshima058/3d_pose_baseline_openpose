# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 00:14:31 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 03:50:30 2019

@author: Administrator
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as  optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import os
import sys
import csv
import numpy as np

from src.model import LinearModel, weight_init
from src.datasets.cmu_panoptic import CMU

from src import Bar
import src.utils as utils
import time
from opt import Options
from src.procrustes import get_transformation
import src.data_process as data_process
from src.poseVisualizer import visualizePose

#from create_data_norm import calcu_mean_std
torch.manual_seed(1)


def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr
    
    # create model
    print(">>> creating model")
    model = LinearModel(joint_num=opt.joint_num)
   
    model=model.cuda()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(reduction='elementwise_mean').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

   # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        start_epoch = ckpt['epoch']
        lr_now = ckpt['lr']
        glob_step = ckpt['step']
        err_best = ckpt['err']
        mean_pose = ckpt['mean_pose']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    
    # data loading
    print(">>> loading data")
    train_data = torch.load(opt.data_dir+'_train.pth')
    mean_pose = np.mean(train_data['tgt'], axis=0)
    mean_pose = np.reshape(mean_pose, (opt.joint_num, 3))
    test_data=CMU(data_path=opt.data_dir+'_test.pth',  use_hg=opt.use_hg)
    train_data=CMU(data_path=opt.data_dir+'_train.pth',  use_hg=opt.use_hg)

    test_loader = DataLoader(dataset=test_data, batch_size=opt.test_batch, shuffle=False,num_workers=opt.job)
    train_loader = DataLoader(dataset=train_data, batch_size=opt.train_batch, shuffle=True,num_workers=opt.job)
    print(">>> data loaded !")
    
    cudnn.benchmark = True
    for epoch in range(start_epoch, opt.epochs):
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch, lr_now))
        
        # train
        glob_step, lr_now, loss_train = train(
            train_loader, model, criterion, optimizer, opt.joint_num, 
            lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
            max_norm=opt.max_norm)
        print("loss_train:", loss_train)

        # test
        outputs_use, loss_test, err_test = test(test_loader, model, criterion, opt.joint_num, procrustes=opt.procrustes)
        
        # save best checkpoint
        is_best = err_test < err_best
        err_best = min(err_test, err_best)
        save_path = ''
        if is_best:
            print("Saved Check Point (error : {})".format(err_test))
            checkpoint = {'epoch': epoch,
                            'lr': lr_now,
                            'step': glob_step,
                            'err': err_best,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'mean_pose': mean_pose}
            if save_path != '' and os.path.exists(save_path):
                os.remove(save_path)
            save_path = opt.ckpt+'_best.chkpt'
            torch.save(checkpoint, save_path)
        
        # write loss to log file
        log_train_file = opt.log + 'train.log'
        with open(log_train_file, 'a') as log_tr:
            log_tr.write('{},{},{},{},{}\n'.format(epoch, lr_now, loss_train, loss_test, err_test))

def train(train_loader, model, criterion, optimizer, joint_num,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          max_norm=True):
    losses = utils.AverageMeter()

    model.train()
 
    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(train_loader))
    
    for i, data in enumerate(train_loader):
        # Turn down Learning Rate
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)
        
        joint2d, truth = data['joint2d'], data['truth']
        inputs=Variable(joint2d.cuda().type(torch.cuda.FloatTensor))
        targets=Variable(truth.cuda().type(torch.cuda.FloatTensor))
    
        outputs = model(inputs)
        outputs=torch.reshape(outputs,(-1,(joint_num)*3))
        targets=torch.reshape(targets,(-1,(joint_num)*3))

        # calculate loss
        optimizer.zero_grad()
        loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        loss.backward()
        
        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.4f}' \
            .format(batch=i + 1,
                    size=len(train_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()

    bar.finish()
    return glob_step, lr_now, losses.avg


def test(test_loader, model, criterion, joint_num, procrustes=False):
    losses = utils.AverageMeter()

    model.eval()
   
    all_dist = []
    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(test_loader))

    for i, data in enumerate(test_loader):
        joint2d,truth=data['joint2d'],data['truth']
        
        inputs=Variable(joint2d.cuda().type(torch.cuda.FloatTensor))
        targets=Variable(truth.cuda().type(torch.cuda.FloatTensor))

        outputs = model(inputs)

        outputs=torch.reshape(outputs,(-1,(joint_num)*3))
        targets=torch.reshape(targets,(-1,(joint_num)*3))

        # calculate loss
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
    
        sqerr = (outputs - targets) ** 2
        distance = np.zeros((sqerr.shape[0],joint_num+1))
        dist_idx = 0
        for k in np.arange(0, (joint_num+1) * 3, 3):
            distance[:, dist_idx] = torch.sqrt(torch.sum(sqerr[:, k:k + 3], axis=1)).to('cpu').detach().numpy()
            dist_idx += 1
        all_dist.append(distance)
            
        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
            .format(batch=i + 1,
                    size=len(test_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()
        
    all_dist = np.vstack(all_dist)
#    joint_err = np.mean(all_dist, axis=0)
    ttl_err = np.mean(all_dist)
    bar.finish()
    print (">>> error: {} <<<".format(ttl_err))
    
    return targets, losses.avg, ttl_err

if __name__ == "__main__":
   
    option = Options().parse()
    main(option)

