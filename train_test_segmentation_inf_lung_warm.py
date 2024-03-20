# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 20:11:53 2024

@author: fares.bougourzi
"""


import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torchvision.transforms.functional as TF
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
import os.path

import os
import ml_collections

import cv2

import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--ModelName', type=str,
                    default='PYNetworks', choices=['UCTransNet', 'MISSFormer', 'MTUnet', 'SwinUnet'], help='experiment_name')
#, choices=['UCTransNet', 'MISSFormer', 'MTUnet', 'SwinUnet']TrAttUnetSI
# parser.add_argument('--ModelNameSave', type=str,
#                     default='UCTransNet', help='experiment_name')
parser.add_argument('--device', type=int,
                    default=1, help='input patch size of network input')
parser.add_argument('--dataset', type=str,
                    default='data2', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--runs', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--itrsave', type=int,
                    default=0, help='output channel of network')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--max_epochs', type=int,
                    default=60, help='maximum epoch number to train')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--batch_size', type=int,
                    default=24, help='input patch size of network input')

parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used') 

args = parser.parse_args()

args.ModelNameSave = 'PYAttUNet' 
dataset_idx = ''
# choices=['UNet', 'UNetplus', 'AttUnet', 'PAttUnet']


############################
#############################    
############################
#############################    
##########################################   
from torch.utils.data import Dataset
########################################## 
##########################################   
class Data_loader2(Dataset):
    def __init__(self, root, train, transform=None):

        self.train = train  # training set or test set
        self.data = np.load(os.path.join(root, train + "_Slice.npy"))
        self.y = np.load(os.path.join(root, train + "_Inf.npy"))
        self.yy = np.load(os.path.join(root, train + "_Lung.npy"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, y, yy = self.data[index], self.y[index], self.yy[index]      
        y[y > 0.0] = 1.0
        yy[yy > 0.0] = 1.0
              
        if self.transform is not None:
            augmentations = self.transform(image=img, mask=y, mask0 = yy)
            image = augmentations["image"]
            mask = augmentations["mask"]
            mask1 = augmentations["mask0"]
            
        return   image, mask, mask1

    def __len__(self):
        return len(self.data)
##########################################  
############################# 
############################
train_transform = A.Compose(
    [
        A.Resize(height=args.img_size, width=args.img_size),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],additional_targets={'mask0': 'mask'}
)
######
val_transform = A.Compose(
    [
        A.Resize(height=args.img_size, width=args.img_size),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],additional_targets={'mask0': 'mask'}
)

############################
############################
############################# 

# train_set = Data_loaderV(
#         root='./Datas'
#         ,train = 'Train_' + dataset_idx +'.pt'
#         ,transform = train_transform 
# )

# validate_set = Data_loaderV(
#         root='./Datas'
#         ,train = 'Test_' + dataset_idx +'.pt'
#         ,transform = val_transforms
# )

train_set = Data_loader2(
        root='./Datas/'
        ,train = 'Train'
        ,transform = train_transform
)

validate_set = Data_loader2(
        root='./Datas/'
        ,train = 'Val'
        ,transform = val_transform
)


###############################################
###### Losses #################################
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

from torch.nn.modules.loss import CrossEntropyLoss

########################################################
#######################################################









##########################################################################
# Models
##########################################################################
# UCTransUnet
if args.ModelName == 'UCTransNet':
    def get_CTranS_config():
        config = ml_collections.ConfigDict()
        config.transformer = ml_collections.ConfigDict()
        config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
        config.transformer.num_heads  = 4
        config.transformer.num_layers = 4
        config.expand_ratio           = 4  # MLP channel dimension expand ratio
        config.transformer.embeddings_dropout_rate = 0
        config.transformer.attention_dropout_rate = 0
        config.transformer.dropout_rate = 0
        config.patch_sizes = [16,8,4,2]
        config.base_channel = 64 # base channel of U-Net
        config.n_classes = 2
        return config
    config = get_CTranS_config()
    
elif  args.ModelName == 'MissFormer':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", default=224)
    parser.add_argument("--num_classes", default=args.num_classes)
    parser.add_argument("--patches_size", default=16)
    parser.add_argument("--n_skip", default=1)
    config  = parser.parse_args()    
    
# elif  args.ModelName == 'MissFormer':
#     from MISSFormernetworks.MISSFormer import MISSFormer
#     model = MISSFormer(num_classes=args.num_classes)
elif  args.ModelName == 'SwinUnet':   
    from networksSwin.config import get_config
    config = get_config(args)
    
####################################
####################################

criterion = CrossEntropyLoss()

###############################################
##############################################

import numpy as np
from medpy import metric

def calculate_hd95(prediction, label, nclasses):
    def calculate_metric_percase(pred, gt):
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0 and gt.sum()>0:
            hd95 = metric.binary.hd95(pred, gt)
            return hd95
        elif pred.sum() > 0 and gt.sum()==0:
            return 0
        else:
            return 0
        
    metric_list = []    
    for i in range(1, nclasses):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))  
        
    return metric_list

########################################################

###############################################
##############################################
from torch.optim.lr_scheduler import LambdaLR
import math
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
###############################################
############################################## 


########################################################



F1_mean, dise_mean, IoU_mean = [], [], []

for itr in range(args.runs):
    
    
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    train_dise, valid_dise = [], []
    train_dise2, valid_dise2 = [], []
    
    train_IoU, valid_IoU = [], []
    
    train_F1score, valid_F1score = [], []
    
    train_Spec, valid_Spec = [], []
    train_Sens, valid_Sens = [], []
    train_Prec, valid_Prec = [], []    
    
    itr += args.itrsave
    model_sp = "./Models" + str(dataset_idx)+"/" + args.ModelNameSave + "/Models"
    if not os.path.exists(model_sp):
        os.makedirs(model_sp)
    ############################
    
    name_model_final = model_sp+ '/' + str(itr) + '_fi.pt'
    name_model_bestF1 =  model_sp+ '/' + str(itr) + '_bt.pt'
    
    model_spR = "./Models" + str(dataset_idx)+"/" + args.ModelNameSave + "/Results"
    if not os.path.exists(model_spR):
        os.makedirs(model_spR)
        
    training_tsx = model_spR+ '/' + str(itr) + '.txt' 
    
  
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    
    
    
    ######### 
    if args.ModelName == 'UCTransNet':
        import UCTansNet.UCTransNet as networks
        model = getattr(networks, args.ModelName)(config, n_channels=3, n_classes=args.num_classes, img_size=args.img_size)
    elif  args.ModelName == 'MISSFormer':
        from MISSFormernetworks.MISSFormer import MISSFormer
        model = MISSFormer(num_classes=args.num_classes)
    elif  args.ModelName == 'MTUnet' :
        from MTUnetmodel.MTUNet import MTUNet
        model = MTUNet(args.num_classes) 
    elif  args.ModelName == 'SwinUnet' :
        from networksSwin.vision_transformer import SwinUnet as ViT_seg 
        model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes)
    elif args.ModelName == 'UNETR':
        print('done')
        import TransUnetArch4AttY as networks
        model = getattr(networks, args.ModelName)(in_channels=3, out_channels= args.num_classes, img_size = args.img_size)
        
    elif args.ModelName == 'PYNetworks':
        print('done2')
        import PYNetworks as networks
        model = getattr(networks, args.ModelNameSave)(input_channels=3, num_classes = args.num_classes)

    else:

        
        # import TransUnetArch4AttSI as networks
        modl_nn = 'UNETR'
        # model = getattr(networks, modl_nn)(in_channels=3, out_channels=1, img_size=224) 
        # name3  = 'C:\\Fares\\2023 AIR revision\\Models2\\TrAttUnet\\Models\\0_bt.pt'
        # model.load_state_dict(torch.load(name3))
        # model.final11 = nn.Conv2d(32, 3, kernel_size=1)
        #######   
        import TransUnetArch4AttSILHSI2Nconv11 as networks
        model = getattr(networks, modl_nn)(in_channels=3, out_channels=3, img_size=224)

          
     

    
    torch.set_grad_enabled(True)    
    ############################
    # Part 5
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=30, shuffle=False)
    
    divs = "cuda:"+str(args.device)
    device = torch.device(divs)
    
    start = time.time()
    model.to(device)
    
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    train_dise1,train_dise2, valid_dise1, valid_dise2 = [], [], [], []
    train_IoU1, train_IoU2, valid_IoU1, valid_IoU2  = [], [], [], []
    train_HD95, train_HD951, train_HD952, valid_HD95, valid_HD951, valid_HD952  = [], [], [], [], [], []
    
    train_F1score1, train_F1score2, valid_F1score1, valid_F1score2 = [], [], [], []
        
    epoch_count = []
    
    best_F1score = -1
    epochs = args.max_epochs
    iter_num = -1

    LR = args.base_lr
    t_total = len(train_loader)* args.max_epochs 
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    scheduler = WarmupCosineSchedule(optimizer, 1*len(train_loader), t_total=t_total)    
    
   
    
    
    for epoch in range(args.max_epochs):
        epoch_count.append(epoch)
             
                
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_loader
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = validate_loader
    
            running_loss = 0.0
    
            num_correct = 0
            num_pixels = 0
    
            step = 0
    
            # iterate over data
            dice_scores = 0
            dice_scores2 = 0
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for batch in tqdm(dataloader):
                x,y,y2 = batch
                x = x.to(device)
                y = y.long().to(device)
                y2 = y2.long().to(device)
       
                step += 1
    
                # forward pass
                if phase == 'train':
                    # zero the gradients
                    outputs, outputs2 = model(x)
                    # calculate the loss 
                    loss1 = criterion(outputs.squeeze(dim=1), y)
                    loss2 = criterion(outputs2.squeeze(dim=1), y2)
                    loss = 0.7*loss1 + 0.3*loss2
                    optimizer.zero_grad()
                    loss.backward()
                    scheduler.step()
                    optimizer.step()
                
                
                else:
                    with torch.no_grad():
                        outputs, outputs2 = model(x)
                        loss1 = criterion(outputs.squeeze(dim=1), y)
                        loss2 = criterion(outputs2.squeeze(dim=1), y2)
                        loss = 0.7*loss1 + 0.3*loss2
    
                running_loss += loss
                
                preds =torch.argmax(outputs, dim = 1)
                preds = preds.squeeze(dim=1).cpu().numpy().astype(int)
                yy = y > 0.5
                yy = yy.squeeze(dim=1).cpu().numpy().astype(int)
    
                num_correct += np.sum(preds == yy)
    
                TP += np.sum(((preds == 1).astype(int) +
                             (yy == 1).astype(int)) == 2)
                TN += np.sum(((preds == 0).astype(int) +
                             (yy == 0).astype(int)) == 2)
                FP += np.sum(((preds == 1).astype(int) +
                             (yy == 0).astype(int)) == 2)
                FN += np.sum(((preds == 0).astype(int) +
                             (yy == 1).astype(int)) == 2)
                num_pixels += preds.size
                for idice in range(preds.shape[0]):
                    dice_scores += (2 * (preds[idice] * yy[idice]).sum()) / (
                        (preds[idice] + yy[idice]).sum() + 1e-8
                    )
    
                predss = np.logical_not(preds).astype(int)
                yyy = np.logical_not(yy).astype(int)
                for idice in range(preds.shape[0]):
                    dice_sc1 = (2 * (preds[idice] * yy[idice]).sum()) / (
                        (preds[idice] + yy[idice]).sum() + 1e-8
                    )
                    dice_sc2 = (2 * (predss[idice] * yyy[idice]).sum()) / (
                        (predss[idice] + yyy[idice]).sum() + 1e-8
                    )
                    dice_scores2 += (dice_sc1 + dice_sc2) / 2
    
                del x
                del y
    
            epoch_loss = running_loss / len(dataloader.dataset)
    
            epoch_acc2 = (num_correct/num_pixels)*100
            epoch_dise = dice_scores/len(dataloader.dataset)
            epoch_dise2 = dice_scores2/len(dataloader.dataset)
    
            Spec = 1 - (FP/(FP+TN))
            Sens = TP/(TP+FN)  # Recall
            Prec = TP/(TP+FP + 1e-8)
            # F1score = 2 *(Sens*Prec) / (Sens+Prec+ 1e-8)
            F1score = TP / (TP + ((1/2)*(FP+FN)) + 1e-8)
            IoU = TP / (TP+FP+FN)
    
            if phase == 'valid':
                if F1score > best_F1score:
                    best_F1score = F1score
                    torch.save(model.state_dict(), name_model_bestF1)
    
    
            with open(training_tsx, "a") as f:
              # print(model, file=f)                      
                # print( 'Epoch', epoch, file=f)
                
                print('Epoch {}/{}'.format(epoch, epochs - 1), file=f)
                print('-' * 10, file=f)                
                print('{} Loss: {:.4f} Acc: {:.8f} Dise: {:.8f} Dise2: {:.8f} IoU: {:.8f} F1: {:.8f} Spec: {:.8f} Sens: {:.8f} Prec: {:.8f}'
                      .format(phase, epoch_loss, epoch_acc2, epoch_dise, epoch_dise2, IoU, F1score, Spec, Sens, Prec), file=f)
    
            train_loss.append(np.array(epoch_loss.detach().cpu())) if phase == 'train' \
                else valid_loss.append(np.array(epoch_loss.detach().cpu()))
            train_acc.append(np.array(epoch_acc2)) if phase == 'train' \
                else valid_acc.append((np.array(epoch_acc2)))
            train_dise.append(np.array(epoch_dise)) if phase == 'train' \
                else valid_dise.append((np.array(epoch_dise)))
            train_dise2.append(np.array(epoch_dise2)) if phase == 'train' \
                else valid_dise2.append((np.array(epoch_dise2)))
    
            train_IoU.append(np.array(IoU)) if phase == 'train' \
                else valid_IoU.append((np.array(IoU)))
    
            train_F1score.append(np.array(F1score)) if phase == 'train' \
                else valid_F1score.append((np.array(F1score)))
    
            train_Spec.append(np.array(Spec)) if phase == 'train' \
                else valid_Spec.append((np.array(Spec)))
            train_Sens.append(np.array(Sens)) if phase == 'train' \
                else valid_Sens.append((np.array(Sens)))
            train_Prec.append(np.array(Prec)) if phase == 'train' \
                else valid_Prec.append((np.array(Prec)))
    
    torch.save(model.state_dict(), name_model_final)
    time_elapsed = time.time() - start
    with open(training_tsx, "a") as f:
      # print(model, file=f)     
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), file=f)
        
    
    
    ############################
    with open(training_tsx, "a") as f:
        
        print('Train', file=f)
        print('Train F1 score', file=f)
        print(train_F1score[valid_F1score.index(np.max(valid_F1score))], file=f)
        
        print(train_acc[valid_F1score.index(np.max(valid_F1score))], file=f)
        print(train_dise[valid_F1score.index(np.max(valid_F1score))], file=f)
        print(train_dise2[valid_F1score.index(np.max(valid_F1score))], file=f)
        print(train_IoU[valid_F1score.index(np.max(valid_F1score))], file=f)
        print(train_Sens[valid_F1score.index(np.max(valid_F1score))], file=f)
        print(train_Spec[valid_F1score.index(np.max(valid_F1score))], file=f)
        print(train_Prec[valid_F1score.index(np.max(valid_F1score))], file=f)
    
        print('-' * 10, file=f)
        print('train Results', file=f)
        print('train_acc', train_acc, file=f)
        print('train_F1score', train_F1score, file=f)
        print('train_dise',train_dise, file=f)
        print('train_IoU',train_IoU, file=f)
        print('train_Sens',train_Sens, file=f)
        print('train_Spec',train_Spec, file=f)
        print('train_Prec',train_Prec, file=f)
         
        print('-' * 10, file=f)
        print(np.max(valid_dise), file=f)
        print('Train', file=f)
        print('Best Val F1 score', file=f)
        print(np.max(valid_F1score), file=f)
        print('Index of Best', file=f)
        print(name_model_bestF1, file=f)
        print(valid_F1score.index(np.max(valid_F1score)), file=f)    
        
        print('-' * 10, file=f)
        print('Val Results', file=f)
        print('valid_acc', valid_acc[valid_F1score.index(np.max(valid_F1score))], file=f)
        print('valid_F1score', valid_F1score[valid_F1score.index(np.max(valid_F1score))], file=f)
        print('valid_dise',valid_dise[valid_F1score.index(np.max(valid_F1score))], file=f)
        print('valid_IoU',valid_IoU[valid_F1score.index(np.max(valid_F1score))], file=f)
        print('valid_Sens',valid_Sens[valid_F1score.index(np.max(valid_F1score))], file=f)
        print('valid_Spec',valid_Spec[valid_F1score.index(np.max(valid_F1score))], file=f)
        print('valid_Prec',valid_Prec[valid_F1score.index(np.max(valid_F1score))], file=f)
                
        print('-' * 10, file=f)
        print('Val Results', file=f)
        print('valid_acc', valid_acc, file=f)
        print('valid_F1score', valid_F1score, file=f)
        print('valid_dise',valid_dise, file=f)
        print('valid_IoU',valid_IoU, file=f)
        print('valid_Sens',valid_Sens, file=f)
        print('valid_Spec',valid_Spec, file=f)
        print('valid_Prec',valid_Prec, file=f) 
        F1_mean.append(valid_F1score[valid_F1score.index(np.max(valid_F1score))])
        dise_mean.append(valid_dise[valid_F1score.index(np.max(valid_F1score))]) 
        IoU_mean.append(valid_IoU[valid_F1score.index(np.max(valid_F1score))])        
                
    f.close()
    
    
std1 = np.std(F1_mean[:5])
std2 = np.std(dise_mean[:5])
std3 = np.std(IoU_mean[:5])

training_tsx = model_spR + '/' + 'mean2' + '.txt'
F1_mean.append(np.mean(F1_mean[:5]))
dise_mean.append(np.mean(dise_mean[:5]))
IoU_mean.append(np.mean(IoU_mean[:5]))


F1_mean.append(std1)
dise_mean.append(std2)
IoU_mean.append(std3)
with open(training_tsx, "a") as f:

    print('F1_mean', F1_mean, file=f)
    print('dise_mean', dise_mean, file=f)
    print('IoU_mean', IoU_mean, file=f)


f.close()
