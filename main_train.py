#!/usr/bin/python3
# these code is for ISIC 2018: Skin Lesion Analysis Towards Melanoma Segmentation
# -*- coding: utf-8 -*-
# @Author  : Duwei Dai
import os
import torch
import math
import visdom
import torch.utils.data as Data
import argparse
import numpy as np
import sys
from tqdm import tqdm

from distutils.version import LooseVersion
from Datasets.ISIC2018 import ISIC2018_dataset
from utils.transform import ISIC2018_transform, ISIC2018_transform_320, ISIC2018_transform_newdata

from Models.networks.ca_network import Comprehensive_Atten_Unet
from Models.DeepLabV3Plus.network import deeplabv3plus_resnet50
from Models.denseaspp.models.DenseASPP_ddw import DenseASPP_121
from Models.compare_networks.BCDU_Net import BCDU_net_D3
from Models.compare_networks.CPFNet import CPF_Net
from Models.compare_networks.CENet import CE_Net
from Models.networks.ms_red_models import Ms_red_v1, Ms_red_v2

from utils.dice_loss import get_soft_label, val_dice_isic, SoftDiceLoss
from utils.dice_loss import Intersection_over_Union_isic
from utils.dice_loss_github import SoftDiceLoss_git, CrossentropyND

from utils.evaluation import AverageMeter
from utils.binary import assd, dc, jc, precision, sensitivity, specificity, F1, ACC
from torch.optim import lr_scheduler

from time import *


Test_Model = {
              "Comp_Atten_Unet": Comprehensive_Atten_Unet,
              "BCDU_net_D3": BCDU_net_D3,
              "CPF_Net": CPF_Net,
              "CE_Net": CE_Net,
              "Ms_red_v1":Ms_red_v1,
              "Ms_red_v2":Ms_red_v2
             }
             
             
Test_Dataset = {'ISIC2018': ISIC2018_dataset}

Test_Transform = {'A': ISIC2018_transform, 'B':ISIC2018_transform_320, "C":ISIC2018_transform_newdata}

criterion = "loss_C"  # loss_A-->SoftDiceLoss;  loss_B-->softdice;  loss_C--> CE + softdice


class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass 
        
        
def train(train_loader, model, criterion, scheduler, optimizer, args, epoch):
    losses = AverageMeter()
    # current_loss_f = "CE_softdice"       # softdice or CE_softdice
    
    model.train()
    for step, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        image = x.float().cuda()                                   
        target = y.float().cuda()                                  

        output = model(image)                                     
        target_soft_a = get_soft_label(target, args.num_classes)   
        target_soft = target_soft_a.permute(0, 3, 1, 2)           

        ca_soft_dice_loss = SoftDiceLoss()
        soft_dice_loss = SoftDiceLoss_git(batch_dice=True, dc_log=False)
        soft_dice_loss2 = SoftDiceLoss_git(batch_dice=False, dc_log=False)
        soft_dice_loss3 = SoftDiceLoss_git(batch_dice=True, dc_log=True)
        CE_loss_F = CrossentropyND()
        
        if criterion == "loss_A":
            loss_ave, loss_lesion = ca_soft_dice_loss(output, target_soft_a, args.num_classes)     
            loss = loss_ave
        
        if criterion == "loss_B":
            dice_loss = soft_dice_loss(output, target_soft)      
            loss = dice_loss
        
        if criterion == "loss_C":
            dice_loss = soft_dice_loss2(output, target_soft)    
            ce_loss = CE_loss_F(output, target)
            loss = dice_loss + ce_loss  
                 
        loss = loss
        losses.update(loss.data, image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % (math.ceil(float(len(train_loader.dataset))/args.batch_size)) == 0:
                   print('current lr: {} | Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {losses.avg:.6f}'.format(
                   optimizer.state_dict()['param_groups'][0]['lr'],
                   epoch, step * len(image), len(train_loader.dataset),
                   100. * step / len(train_loader), losses=losses))
                
    print('The average loss:{losses.avg:.4f}'.format(losses=losses))
    return losses.avg


def valid_isic(valid_loader, model, criterion, optimizer, args, epoch, best_score, val_acc_log):
    isic_Jaccard = []
    isic_dc = []

    model.eval()
    for step, (t, k) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        image = t.float().cuda()
        target = k.float().cuda()

        output = model(image)                                             # model output
        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_dis_test = output_dis.permute(0, 2, 3, 1).float()
        target_test = target.permute(0, 2, 3, 1).float()
        isic_b_Jaccard = jc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        isic_b_dc = dc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        isic_Jaccard.append(isic_b_Jaccard)
        isic_dc.append(isic_b_dc)

    isic_Jaccard_mean = np.average(isic_Jaccard)
    isic_dc_mean = np.average(isic_dc)
    net_score = isic_Jaccard_mean + isic_dc_mean
        
    print('The ISIC Dice score: {dice: .4f}; '
          'The ISIC JC score: {jc: .4f}'.format(
           dice=isic_dc_mean, jc=isic_Jaccard_mean))
           
    with open(val_acc_log, 'a') as vlog_file:
        line = "{} | {dice: .4f} | {jc: .4f}".format(epoch, dice=isic_dc_mean, jc=isic_Jaccard_mean)
        vlog_file.write(line+'\n')

    if net_score > max(best_score):
        best_score.append(net_score)
        print(best_score)
        modelname = args.ckpt + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
        print('the best model will be saved at {}'.format(modelname))
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
        torch.save(state, modelname)

    return isic_Jaccard_mean, isic_dc_mean, net_score


def test_isic(test_loader, model, num_para, args, test_acc_log):
    isic_dice = []
    isic_iou = []
   # isic_assd = []
    isic_acc = []
    isic_sensitive = []
    isic_specificy = []
    isic_precision = []
    isic_f1_score = []
    isic_Jaccard_M = []
    isic_Jaccard_N = []
    isic_Jaccard = []
    isic_dc = []
    infer_time = []
    
    modelname = args.ckpt + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    model.eval()
    for step, (name, img, lab) in tqdm(enumerate(test_loader), total=len(test_loader)):
        image = img.float().cuda()
        target = lab.float().cuda() # [batch, 1, 224, 320]
        
        begin_time = time()
        output = model(image)
        end_time = time()
        pred_time = end_time - begin_time
        infer_time.append(pred_time)
        
        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_dis_test = output_dis.permute(0, 2, 3, 1).float()
        target_test = target.permute(0, 2, 3, 1).float()
        output_soft = get_soft_label(output_dis, 2) 
        target_soft = get_soft_label(target, 2) 

        label_arr = np.squeeze(target_soft.cpu().numpy()).astype(np.uint8)
        output_arr = np.squeeze(output_soft.cpu().byte().numpy()).astype(np.uint8)

        isic_b_dice = val_dice_isic(output_soft, target_soft, 2)                                         # the dice
        isic_b_iou = Intersection_over_Union_isic(output_dis_test, target_test, 1)                       # the iou
        # isic_b_asd = assd(output_arr[:, :, 1], label_arr[:, :, 1])                                     # the assd
        isic_b_acc = ACC(output_dis_test.cpu().numpy(), target_test.cpu().numpy())                       # the accuracy
        isic_b_sensitive = sensitivity(output_dis_test.cpu().numpy(), target_test.cpu().numpy())         # the sensitivity
        isic_b_specificy = specificity(output_dis_test.cpu().numpy(), target_test.cpu().numpy())         # the specificity
        isic_b_precision = precision(output_dis_test.cpu().numpy(), target_test.cpu().numpy())           # the precision
        isic_b_f1_score = F1(output_dis_test.cpu().numpy(), target_test.cpu().numpy())                   # the F1
        isic_b_Jaccard_m = jc(output_arr[:, :, 1], label_arr[:, :, 1])                                   # the Jaccard melanoma
        isic_b_Jaccard_n = jc(output_arr[:, :, 0], label_arr[:, :, 0])                                   # the Jaccard no-melanoma
        isic_b_Jaccard = jc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        isic_b_dc = dc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        
        dice_np = isic_b_dice.data.cpu().numpy()
        iou_np = isic_b_iou.data.cpu().numpy()
       
        isic_dice.append(dice_np)
        isic_iou.append(iou_np)
       # isic_assd.append(isic_b_asd)
        isic_acc.append(isic_b_acc)
        isic_sensitive.append(isic_b_sensitive)
        isic_specificy.append(isic_b_specificy)
        isic_precision.append(isic_b_precision)
        isic_f1_score.append(isic_b_f1_score)
        isic_Jaccard_M.append(isic_b_Jaccard_m)
        isic_Jaccard_N.append(isic_b_Jaccard_n)
        isic_Jaccard.append(isic_b_Jaccard)
        isic_dc.append(isic_b_dc)
        
        with open(test_acc_log, 'a') as tlog_file:
            line = "{} | {dice: .4f} | {jc: .4f}".format(name[0], dice=isic_b_dc, jc=isic_b_Jaccard)
            tlog_file.write(line+'\n')

    all_time = np.sum(infer_time)
    isic_dice_mean = np.average(isic_dice)
    isic_dice_std = np.std(isic_dice)

    isic_iou_mean = np.average(isic_iou)
    isic_iou_std = np.std(isic_iou)

   # isic_assd_mean = np.average(isic_assd)
   # isic_assd_std = np.std(isic_assd)
      
    isic_acc_mean = np.average(isic_acc)
    isic_acc_std = np.std(isic_acc)
    
    isic_sensitive_mean = np.average(isic_sensitive)
    isic_sensitive_std = np.std(isic_sensitive)
    
    isic_specificy_mean = np.average(isic_specificy)
    isic_specificy_std = np.std(isic_specificy)
    
    isic_precision_mean = np.average(isic_precision)
    isic_precision_std = np.std(isic_precision)
    
    isic_f1_score_mean = np.average(isic_f1_score)
    iisic_f1_score_std = np.std(isic_f1_score)
    
    isic_Jaccard_M_mean = np.average(isic_Jaccard_M)
    isic_Jaccard_M_std = np.std(isic_Jaccard_M)
    
    isic_Jaccard_N_mean = np.average(isic_Jaccard_N)
    isic_Jaccard_N_std = np.std(isic_Jaccard_N)
    
    isic_Jaccard_mean = np.average(isic_Jaccard)
    isic_Jaccard_std = np.std(isic_Jaccard)
    
    isic_dc_mean = np.average(isic_dc)
    isic_dc_std = np.std(isic_dc)

    print('The ISIC mean dice: {isic_dice_mean: .4f}; The ISIC dice std: {isic_dice_std: .4f}'.format(
           isic_dice_mean=isic_dice_mean, isic_dice_std=isic_dice_std))
    print('The ISIC mean IoU: {isic_iou_mean: .4f}; The ISIC IoU std: {isic_iou_std: .4f}'.format(
           isic_iou_mean=isic_iou_mean, isic_iou_std=isic_iou_std))
   # print('The ISIC mean assd: {isic_assd_mean: .4f}; The ISIC assd std: {isic_assd_std: .4f}'.format(
   #        isic_assd_mean=isic_assd_mean, isic_assd_std=isic_assd_std))
    print('The ISIC mean ACC: {isic_acc_mean: .4f}; The ISIC ACC std: {isic_acc_std: .4f}'.format(
           isic_acc_mean=isic_acc_mean, isic_acc_std=isic_acc_std))
    print('The ISIC mean sensitive: {isic_sensitive_mean: .4f}; The ISIC sensitive std: {isic_sensitive_std: .4f}'.format(
           isic_sensitive_mean=isic_sensitive_mean, isic_sensitive_std=isic_sensitive_std)) 
    print('The ISIC mean specificy: {isic_specificy_mean: .4f}; The ISIC specificy std: {isic_specificy_std: .4f}'.format(
           isic_specificy_mean=isic_specificy_mean, isic_specificy_std=isic_specificy_std))
    print('The ISIC mean precision: {isic_precision_mean: .4f}; The ISIC precision std: {isic_precision_std: .4f}'.format(
           isic_precision_mean=isic_precision_mean, isic_precision_std=isic_precision_std))
    print('The ISIC mean f1_score: {isic_f1_score_mean: .4f}; The ISIC f1_score std: {iisic_f1_score_std: .4f}'.format(
           isic_f1_score_mean=isic_f1_score_mean, iisic_f1_score_std=iisic_f1_score_std))
    print('The ISIC mean Jaccard_M: {isic_Jaccard_M_mean: .4f}; The ISIC Jaccard_M std: {isic_Jaccard_M_std: .4f}'.format(
           isic_Jaccard_M_mean=isic_Jaccard_M_mean, isic_Jaccard_M_std=isic_Jaccard_M_std))
    print('The ISIC mean Jaccard_N: {isic_Jaccard_N_mean: .4f}; The ISIC Jaccard_N std: {isic_Jaccard_N_std: .4f}'.format(
           isic_Jaccard_N_mean=isic_Jaccard_N_mean, isic_Jaccard_N_std=isic_Jaccard_N_std))
    print('The ISIC mean Jaccard: {isic_Jaccard_mean: .4f}; The ISIC Jaccard std: {isic_Jaccard_std: .4f}'.format(
           isic_Jaccard_mean=isic_Jaccard_mean, isic_Jaccard_std=isic_Jaccard_std))
    print('The ISIC mean dc: {isic_dc_mean: .4f}; The ISIC dc std: {isic_dc_std: .4f}'.format(
           isic_dc_mean=isic_dc_mean, isic_dc_std=isic_dc_std))
    print('The inference time: {time: .4f}'.format(time=all_time))
    print("Number of trainable parameters {0} in Model {1}".format(num_para, args.id))
    
    
def main(args, val_acc_log, test_acc_log):
    best_score = [0]
    start_epoch = args.start_epoch
    print('loading the {0},{1},{2} dataset ...'.format('train', 'test', 'test'))
    trainset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='train', 
                                       with_name=False, transform=Test_Transform[args.transform])
    validset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='test',
                                       with_name=False, transform=Test_Transform[args.transform])
    testset =  Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='test',
                                       with_name=True, transform=Test_Transform[args.transform])

    trainloader = Data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=6)
    validloader = Data.DataLoader(dataset=validset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)
    testloader = Data.DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)
    print('Loading is done\n')

    args.num_input = 3
    args.num_classes = 2
    args.out_size = (224, 320)

    if args.id =="Comp_Atten_Unet":
        model = Test_Model[args.id](args, args.num_input, args.num_classes)

    elif args.id =="DenseASPP_121":
        model = DenseASPP_121(n_class=2, output_stride=8)              
   
    elif args.id =="deeplabv3plus_resnet50":
        model = deeplabv3plus_resnet50(num_classes=2)  
    else:
        model = Test_Model[args.id](classes=2, channels=3)

    model = model.cuda()

    print("------------------------------------------")
    print("Network Architecture of Model {}:".format(args.id))
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul
    print(model)
    print("Number of trainable parameters {0} in Model {1}".format(num_para, args.id))
    print("------------------------------------------")

    # Define optimizers and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate, weight_decay=args.weight_decay)    
   # scheduler = lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.5)                                    # lr_1
   # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = 0.0001)     # lr_2
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = 0.00001)    # lr_3
   # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min = 0.00001)  # lr_4

    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    print("Start training ...")
    for epoch in range(start_epoch+1, args.epochs + 1):
        scheduler.step()
        train_avg_loss = train(trainloader, model, criterion, scheduler, optimizer, args, epoch)
        isic_Jaccard_mean, isic_dc_mean, net_score = valid_isic(validloader, model, criterion, optimizer, args, epoch, best_score, val_acc_log)
        if epoch > args.particular_epoch:
            if epoch % args.save_epochs_steps == 0:
                filename = args.ckpt + '/' + str(epoch) + '_' + args.data + '_checkpoint.pth.tar'
                print('the model will be saved at {}'.format(filename))
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
                torch.save(state, filename)

    print('Training Done! Start testing')
    if args.data == 'ISIC2018':
        test_isic(testloader, model, num_para, args, test_acc_log)
    print('Testing Done!')
    
    
if __name__ == '__main__':

    """
    This project supports the following models:
    compared methods：
        Comp_Atten_Unet:                      ca-net                                  ** 4134 ***
        DenseASPP_121                                                                 ** 6985 ***  
        deeplabv3plus_resnet50:              without pretrain                         ** 5809 ***
        BCDU_net_D3                          batch_size=4, lr_rate = 1e-4
        CPF_Net
        CE_Net

    Ms-RED_famaily:
        Ms_red_v1                                                                     ** 9937 ***  our Ms RED V1
        Ms_red_v2                                                                     ** 10889 *** our Ms RED V2                           
    """
    
    os.environ['CUDA_VISIBLE_DEVICES']= '0'                                                 # gpu-id
    
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'
    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')
    
    parser.add_argument('--id', default="Ms_red_v1",
                        help='Ms_red_v1')                                                   # Select a loaded model name

    # Path related arguments
    parser.add_argument('--root_path', default='/data/ISIC2018_npy_all_224_320',
                        help='root directory of data')                                      # The folder where the numpy data set is stored
    parser.add_argument('--ckpt', default='./saved_models/',
                        help='folder to output checkpoints')                                # The folder in which the trained model is saved
    parser.add_argument('--transform', default='C', type=str,
                        help='which ISIC2018_transform to choose')                         
    parser.add_argument('--data', default='ISIC2018', help='choose the dataset')            
    parser.add_argument('--out_size', default=(224, 320), help='the output image size')
    parser.add_argument('--val_folder', default='folder3', type=str,
                        help='folder1、folder2、folder3、folder4、folder5')                 # five-fold cross-validation

    # optimization related arguments
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',              
                        help='input batch size for training (default: 12)')                 # batch_size
    parser.add_argument('--lr_rate', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')                              
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='weights regularizer')
    parser.add_argument('--particular_epoch', default=30, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=300, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')

    args = parser.parse_args()
    
    args.ckpt = os.path.join(args.ckpt, args.data, args.val_folder, args.id + "_{}".format(criterion))    
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)
    logfile = os.path.join(args.ckpt,'{}_{}_{}.txt'.format(args.val_folder, args.id, criterion))          # path of the training log
    sys.stdout = Logger(logfile)  
    
    val_acc_log = os.path.join(args.ckpt,'val_acc_{}_{}_{}.txt'.format(args.val_folder, args.id, criterion))   
    test_acc_log = os.path.join(args.ckpt,'test_acc_{}_{}_{}.txt'.format(args.val_folder, args.id, criterion))    
    
    print('Models are saved at %s' % (args.ckpt))
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str(args.start_epoch) + '_' + args.data + '_checkpoint.pth.tar'

    main(args, val_acc_log, test_acc_log)
