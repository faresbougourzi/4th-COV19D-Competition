# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:33:55 2024

@author: fares.bougourzi
"""


import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
import numpy as np

from tqdm import tqdm

import os

from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, accuracy_score

import time

from utils.swarm_schedule import  WarmupCosineSchedule
from utils.gt_correct import  get_num_correct
from utils.losses import FocalLoss

from networks import HDeCOVNet 
import importlib


from networks import densenet 
from networks import pre_act_resnet
from networks import resnet
from networks import resnet2p1d
from networks import resnext
from networks import wide_resnet

import warnings
warnings.filterwarnings("ignore")


'''
densenet
pre_act_resnet
resnet
resnet2p1d
resnext
wide_resnet


r2p1d18_K_200ep.pth
r2p1d34_K_200ep.pth
r2p1d50_K_200ep.pth
r2p1d50_KM_200ep.pth
r3d18_K_200ep.pth
r3d18_KM_200ep.pth
r3d34_K_200ep.pth
r3d34_KM_200ep.pth
r3d50_K_200ep.pth
r3d50_KM_200ep.pth
r3d50_KMS_200ep.pth
r3d50_KS_200ep.pth
r3d50_M_200ep.pth
r3d50_MS_200ep.pth
r3d50_S_200ep.pth
r3d101_K_200ep.pth
r3d101_KM_200ep.pth
r3d152_K_200ep.pth
r3d152_KM_200ep.pth
r3d200_K_200ep.pth
r3d200_KM_200ep.pth


r3d18_K_200ep.pth: --model resnet --model_depth 18 --n_pretrain_classes 700
r3d18_KM_200ep.pth: --model resnet --model_depth 18 --n_pretrain_classes 1039
r3d34_K_200ep.pth: --model resnet --model_depth 34 --n_pretrain_classes 700
r3d34_KM_200ep.pth: --model resnet --model_depth 34 --n_pretrain_classes 1039
r3d50_K_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 700
r3d50_KM_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 1039
r3d50_KMS_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 1139
r3d50_KS_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 800
r3d50_M_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 339
r3d50_MS_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 439
r3d50_S_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 100
r3d101_K_200ep.pth: --model resnet --model_depth 101 --n_pretrain_classes 700
r3d101_KM_200ep.pth: --model resnet --model_depth 101 --n_pretrain_classes 1039
r3d152_K_200ep.pth: --model resnet --model_depth 152 --n_pretrain_classes 700
r3d152_KM_200ep.pth: --model resnet --model_depth 152 --n_pretrain_classes 1039
r3d200_K_200ep.pth: --model resnet --model_depth 200 --n_pretrain_classes 700
r3d200_KM_200ep.pth: --model resnet --model_depth 200 --n_pretrain_classes 1039
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ModelName', type=str,
                    default='HDeCOVNet', choices=['densenet','pre_act_resnet', 'resnet','resnet2p1d','resnext','wide_resnet'], help='experiment_name')
parser.add_argument('--Modelversion', type=int,
                    default=18, choices=[10, 18, 34, 50, 101, 152, 200, 1212, 169, 201, 264], help='experiment_name')
parser.add_argument('--n_pretrain_classes', type=int,
                    default=1039, choices=[10, 18, 34, 50, 101, 152, 200, 121, 169, 201, 264], help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--Pretrained', type=bool,
                    default=False, help='experiment_name')
parser.add_argument('--Pretrainedpath', type=str,
                    default='./PretrainedModels/r3d18_KM_200ep.pth', help='experiment_name')
parser.add_argument('--Remark', type=str,
                    default='origi, infection, lung2, testing augmentation, train1', help='experiment_name')
parser.add_argument('--run', type=int,
                    default=1, help='output channel of network')

parser.add_argument('--Resultstx', type=str,
                    default='Results_train1.txt', help='experiment_name')
parser.add_argument('--dataset_idx', type=str,
                    default='3D_Train1', help='experiment_name')
parser.add_argument('--in_ch', type=str,
                    default=3, help='experiment_name')


parser.add_argument('--device', type=int,
                    default=1, help='input patch size of network input')
parser.add_argument('--device_ids', type=int,
                    default=[1, 2], help='input patch size of network input')
parser.add_argument('--num_device', type=int,
                    default= 2, help='input patch size of network input')

parser.add_argument('--runs', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--itrsave', type=int,
                    default=0, help='output channel of network')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--max_epochs', type=int,
                    default=80, help='maximum epoch number to train')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--batch_size', type=int,
                    default=16, help='input patch size of network input')
parser.add_argument('--ts_bz', type=int,
                    default=32, help='input patch size of network input')

parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used') 
args = parser.parse_args()



epochs = args.max_epochs
divs = "cuda:"+str(args.device)
device = torch.device(divs)


epochs = args.max_epochs
divs = "cuda:"+str(args.device)
device = torch.device(divs)

img_size = args.img_size

from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
   
from torch.utils import data
import albumentations as A
train_transforms = A.ReplayCompose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Rotate(limit=40, p=1.0),
        A.VerticalFlip(p=0.2),
        A.VerticalFlip(p=0.2), A.Blur(blur_limit=(3, 3), p=0.2),
        A.MultiplicativeNoise(multiplier=1.5, p=0.2), 
        A.MultiplicativeNoise(multiplier=0.5, p=0.2), 
        A.RandomBrightness(limit=0.2, always_apply=False, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.2),
        A.RandomContrast (limit=0.2, always_apply=False, p=0.2),
        A.RandomGridShuffle (grid=(3, 3), always_apply=False, p=0.2),
        A.RandomToneCurve (scale=0.1, always_apply=False, p=0.2),            
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ], additional_targets={'image0': 'image', 'image1': 'image'}
)
        
test_transforms =     A.ReplayCompose([
        A.Resize(height=img_size, width=img_size),          
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ], additional_targets={'image0': 'image', 'image1': 'image'})

##################################################       
##########################################
##################################################    
class Covid_loader_pt2(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, path, transform=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.path = path
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        pth = self.path
        
        labels = np.load(os.path.join(pth, 'labels.npy'))[ID]
        
        X1  = np.load(os.path.join(pth, str(ID) + '.npy')).astype(np.uint8)    
        X2  = np.load(os.path.join(pth, str(ID) + '_inf.npy')).astype(np.uint8)          
        X3  = np.load(os.path.join(pth, str(ID) + '_lung2.npy')).astype(np.uint8) 
       
        if self.transform is not None:
            Xs1 = []

            data1 = self.transform(image=np.expand_dims(X1[:,:,0],2), image0=X2[:,:,0], image1=X2[:,:,0])

            img1 = data1['image']
            img2 = data1['image0']
            img3 = data1['image1']
            img = torch.concat([img1, img2, img3], dim = 0)
            Xs1.append(img)

            for i in range(X2.shape[2]-1):
                image2_data = A.ReplayCompose.replay(data1['replay'], image=np.expand_dims(X1[:,:,i+1],2), image0=X2[:,:,i+1], image1=X3[:,:,i+1])
                img1 = image2_data['image']
                img2 = image2_data['image0']
                img3 = image2_data['image1']
                img = torch.concat([img1, img2, img3], dim = 0)
                Xs1.append(img)

     
        Xs1 = torch.stack(Xs1).transpose(0, 1).transpose(1, 2)#.squeeze(dim = 1)
        
        Xs1 = F.interpolate(Xs1, [64, 224], mode ='bicubic').transpose(1, 2)
        # print(Xs1.shape)
       
        return Xs1, labels 
#############################################
##################################################    
class Covid_loader_pt3(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, path, transform=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.path = path
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        pth = self.path
        
        labels = np.load(os.path.join(pth, 'labels.npy'))[ID]
        
        X1  = np.load(os.path.join(pth, str(ID) + '.npy')).astype(np.uint8)    
        X2  = np.load(os.path.join(pth, str(ID) + '_inf.npy')).astype(np.uint8)          
        X3  = np.load(os.path.join(pth, str(ID) + '_lung2.npy')).astype(np.uint8) 
               
        rep_aug = []
        
        for ii in range(10):       
            if self.transform is not None:
                Xs1 = []
    
                data1 = self.transform(image=np.expand_dims(X1[:,:,0],2), image0=X2[:,:,0], image1=X2[:,:,0])
    
                img1 = data1['image']
                img2 = data1['image0']
                img3 = data1['image1']
                img = torch.concat([img1, img2, img3], dim = 0)
                Xs1.append(img)
    
                for i in range(X2.shape[2]-1):
                    image2_data = A.ReplayCompose.replay(data1['replay'], image=np.expand_dims(X1[:,:,i+1],2), image0=X2[:,:,i+1], image1=X3[:,:,i+1])
                    img1 = image2_data['image']
                    img2 = image2_data['image0']
                    img3 = image2_data['image1']
                    img = torch.concat([img1, img2, img3], dim = 0)
                    Xs1.append(img)
    
         
            Xs1 = torch.stack(Xs1).transpose(0, 1).transpose(1, 2)
            
            Xs1 = F.interpolate(Xs1, [64, 224], mode ='bicubic').transpose(1, 2)
            rep_aug.append(Xs1)
        
        return torch.stack(rep_aug), labels  


###############################################
##############################################
###############################################
############################################## 
import sys
def get_class(class_name):
    return getattr(sys.modules[__name__], class_name)



Accracy_vl_mean = []
Accracy_ts_mean = []

MaF1_vl_mean = []
MaF1_ts_mean = [] 
 
WeF1_vl_mean = []
WeF1_ts_mean = []


dataset_idx = args.dataset_idx
if args.ModelName == 'HDeCOVNet':   
    modl_n = args.ModelName+ '_' + str(args.batch_size) + '_' + str(args.in_ch)+ '_' + str(args.base_lr) + '_' + str(args.max_epochs)+ '_' + str(args.Pretrained)+ '_' + str(args.Modelversion)+ '_' + str(args.run)
else:
    modl_n = args.ModelName + str(args.Modelversion)+ '_' + str(args.batch_size) + '_' + str(args.in_ch)+ '_' + str(args.base_lr) + '_' + str(args.max_epochs)+ '_' + str(args.Pretrained)    
    # model = densenet.generate_model(model_depth = 121)



def train():
    runs = args.runs
    for itr in range(runs):
        model_sp = "./Models" + dataset_idx +"/" + modl_n + "/Models"
        if not os.path.exists(model_sp):
            os.makedirs(model_sp)
       
        name_model_final = model_sp+ '/' + str(itr) + '_fi.pt'
        name_model_bestF1 =  model_sp+ '/' + str(itr) + '_bt.pt'
        # name_model_bestsw =  model_sp+ '/' + str(itr) + '_swa.pt'
        
        model_spR = "./Models" + dataset_idx +"/" + modl_n + "/Results"
        if not os.path.exists(model_spR):
            os.makedirs(model_spR)
            
        training_tsx = model_spR+ '/' + str(itr) + '.txt' 
        
        
        import datetime
        import sys
    
        Resultstx = args.Resultstx 
        with open(Resultstx, "a") as f:
            print('_' * 50, file=f) 
            print('_' * 50, file=f)
            print('Experiment date is: ', datetime.datetime.now(), file=f)
            print('Experiment date is: ', datetime.datetime.now(), file=f)
            print(sys.argv[0], file=f) 
            print('Architecture Name', args.ModelName + str(args.Modelversion), file=f)
            print('In Ch', args.in_ch, file=f)
            print('base_lr', args.base_lr, file=f)
            print('max_epochs', args.max_epochs, file=f)
            print('batch_size', args.batch_size, file=f)
            print('Pretrained', args.Pretrained, file=f)
            print('save_training', model_sp, file=f) 
            print('Pretrained model', args.Pretrainedpath, file=f) 
            print('Initial Classes number', args.n_pretrain_classes, file=f) 
            print('Remarque', args.Remark, file=f)
            
    
        
        Acc_best = -2
        Acc_bestsw = -2
        epoch_count = []
        Accracy_tr = []
        Accracy_ts = []
        
        AccracyRA_tr = []
        AccracyRA_vl = []
        AccracyRA_sw = []
        
        MaF1_tr = []
        MaF1_vl = []
        MaF1_sw = []
        
        
        torch.set_printoptions(linewidth=120)
        
        tr_path = 'E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Train'
        vl_path = 'E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Val'
        # E:/Fares/4th Covid-19 challenge
        tr_labels = np.load('E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Train/labels.npy')
        vl_labels= np.load('E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Val/labels.npy')
        
        tr_indxs= list(range(len(tr_labels)))
        train_set = Covid_loader_pt2(
                list_IDs = tr_indxs, 
                path = tr_path, 
                transform=train_transforms
        )
        
        vl_indxs= list(range(len(vl_labels)))
        test_set = Covid_loader_pt2(
                list_IDs = vl_indxs, 
                path = vl_path, 
                transform=test_transforms
        )       
                
        
        
    
        # device = torch.device("cuda:0")
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)
        validate_loader = torch.utils.data.DataLoader(test_set, batch_size=args.ts_bz, num_workers=8, pin_memory=False) 
         
        if args.ModelName == 'HDeCOVNet':   
            model = HDeCOVNet.HDeCOVNet(in_ch = args.in_ch)
        elif args.ModelName == 'resnet' :
            model = getattr(get_class(args.ModelName), 'generate_model')(model_depth = args.Modelversion)
            model.fc = nn.Linear(model.fc.in_features, args.n_pretrain_classes)

            
            
        '''
        densenet
        pre_act_resnet
        resnet
        resnet2p1d
        resnext
        wide_resnet ["arch"]
        '''           
        if args.Pretrained:
            model.load_state_dict(torch.load(args.Pretrainedpath)["state_dict"])
            model.fc = nn.Linear(model.fc.in_features, args.num_classes)             
            
        if args.num_device>1:
            model = nn.DataParallel(model, device_ids=args.device_ids)
        model.to(device)
        
        LR = args.base_lr
        t_total = len(train_loader)* args.max_epochs 
        optimizer = torch.optim.Adam(model.parameters(), lr = LR)
        scheduler = WarmupCosineSchedule(optimizer, 1*len(train_loader), t_total=t_total)            
        
        criterion = FocalLoss()    
        
        Acc_best = -2
        
        ##################################  
        for epoch in range(epochs):
            start_time = time.time()
    
            train_loss = 0
            validation_loss = 0
            total_correct_tr = 0
            total_correct_val = 0
            total_correct_tr2 = 0
            total_correct_val2 = 0
           
            label_f1tr = []
            pred_f1tr = []
            
            for batch in tqdm(train_loader):
                images1, labels = batch
                images1 = images1.squeeze(dim=1).float().to(device)
                labels = labels.long().to(device)
                torch.set_grad_enabled(True)
                model.train()
                preds = model(images1)
                loss = criterion(preds, labels)
        
                optimizer.zero_grad()
                loss.backward()
    
                scheduler.step()
                optimizer.step()
        
                train_loss += loss.item()
                total_correct_tr += get_num_correct(preds, labels)
        
                label_f1tr.extend(labels.cpu().numpy().tolist())
                pred_f1tr.extend(preds.argmax(dim=1).tolist())      
            
                del images1
                del labels
            
            label_f1vl = []
            pred_f1vl = []
           
        
            for batch in tqdm(validate_loader):
                images1, labels = batch
                images1 = images1.squeeze(dim=1).float().to(device)
                labels = labels.long().to(device)
        
                model.eval()
                with torch.no_grad():
                    preds = model(images1)
        
                loss = criterion(preds, labels)
        
                validation_loss += loss.item()
                total_correct_val += get_num_correct(preds, labels)
                
        
                label_f1vl.extend(labels.cpu().numpy().tolist())
                pred_f1vl.extend(preds.argmax(dim=1).tolist())
                
                del images1
                del labels
        
            print('Ep: ', epoch, 'AC_tr: ', total_correct_tr/len(train_set), 'AC_ts: ',
                  total_correct_val/len(test_set),'AC_tr2: ', total_correct_tr2/len(train_set), 'AC_ts2: ',
                  total_correct_val2/len(test_set), 'Loss_tr: ', train_loss/len(train_set))
    
            print('MaF1_tr: ', f1_score(label_f1tr, pred_f1tr, average='macro'), 'MaF1_vl: ',
                  f1_score(label_f1vl, pred_f1vl, average='macro'))
            print('CM_tr: ')
            print(confusion_matrix(label_f1tr, pred_f1tr))    
            print('CM_vl: ')
            print(confusion_matrix(label_f1vl, pred_f1vl))
            
            
            print("--- %s seconds ---" % (time.time() - start_time)) 
            print("--- %s minutes ---" % ((time.time() - start_time)/60)) 
            ##################################        
            with open(training_tsx, "a") as f:
                print('Epoch {}/{}'.format(epoch, epochs - 1), file=f)
                print('-' * 10, file=f)             
                print('Ep: ', epoch, 'AC_tr: ', total_correct_tr/len(train_set), 'AC_ts: ',
                      total_correct_val/len(test_set),'AC_tr2: ', total_correct_tr2/len(train_set), 'AC_ts2: ',
                      total_correct_val2/len(test_set), 'Loss_tr: ', train_loss/len(train_set), file=f)            
                print('CM_tr: ', file=f)
                print(confusion_matrix(label_f1tr, pred_f1tr), file=f)    
                print('CM_vl: ', file=f)
                print(confusion_matrix(label_f1vl, pred_f1vl), file=f)
              
                
            with open(training_tsx, "a") as f:
                print('MaF1_tr: ', f1_score(label_f1tr, pred_f1tr, average='macro'), 'MaF1_vl: ',
                      f1_score(label_f1vl, pred_f1vl, average='macro'), file=f) 
                
                    
            Acc_best2 = f1_score(label_f1vl, pred_f1vl, average='macro')    
            if Acc_best2 >= Acc_best:
                Acc_best = Acc_best2
                torch.save(model.module.state_dict(), name_model_bestF1)
                
            AccracyRA_tr.append(balanced_accuracy_score(label_f1tr, pred_f1tr))
            AccracyRA_vl.append(balanced_accuracy_score(label_f1vl, pred_f1vl))
                   
            MaF1_tr.append(f1_score(label_f1tr, pred_f1tr , average='macro'))
            MaF1_vl.append(f1_score(label_f1vl, pred_f1vl , average='macro'))  
            












    
               
        """ Val"""
     
        print(Acc_best) 
        if args.ModelName == 'HDeCOVNet':   
            model = HDeCOVNet.HDeCOVNet(in_ch = args.in_ch)
        else:
            model = getattr(get_class(args.ModelName), 'generate_model')(model_depth = args.Modelversion)
            
        if args.ModelName == 'densenet':
            model.classifier = nn.Linear(model.classifier.in_features, args.num_classes) 
        elif args.ModelName == 'resnet':
            model.fc = nn.Linear(model.fc.in_features, args.num_classes) 
            
        model.load_state_dict(torch.load(name_model_bestF1))     
            
        if args.num_device>1:
            model = nn.DataParallel(model, device_ids=args.device_ids)               
        
        model.to(device) 
        ###########################
        
        label_f1vl = []
        pred_f1vl = [] 
        
        validation_loss = 0
        total_correct_val = 0        
        
        for batch in tqdm(validate_loader):
            images1, labels = batch
            images1 = images1.squeeze(dim=1).float().to(device)
            labels = labels.long().to(device)
    
            model.eval()
            with torch.no_grad():
                preds = model(images1)
            loss = criterion(preds, labels)
        
            validation_loss += loss.item()
            total_correct_val += get_num_correct(preds, labels)
        
            label_f1vl.extend(labels.cpu().numpy().tolist())
            pred_f1vl.extend(preds.argmax(dim=1).tolist())
        
            del images1
            del labels
    
        print('Accuracy', accuracy_score(label_f1vl, pred_f1vl))
        print('F1-score', f1_score(label_f1vl, pred_f1vl, average='macro'))
        print(confusion_matrix(label_f1vl, pred_f1vl))
        
        with open(Resultstx, "a") as f:
            print('-' * 10, file=f) 
            print('Val Results', file=f)         
            print('Accuracy', accuracy_score(label_f1vl, pred_f1vl), file=f)
            print('F1-score', f1_score(label_f1vl, pred_f1vl, average='macro'), file=f)
            print(confusion_matrix(label_f1vl, pred_f1vl), file=f)
            
                      
        """  Test 1, Test 2"""          
            
        tr_path = 'E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Test1'
        vl_path = 'E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Test2'

        tr_labels = np.load('E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Test1/labels.npy')
        vl_labels= np.load('E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Test2/labels.npy')
        
        tr_indxs= list(range(len(tr_labels)))
        train_set = Covid_loader_pt2(
                list_IDs = tr_indxs, 
                path = tr_path, 
                transform=test_transforms
        )
        
        vl_indxs= list(range(len(vl_labels)))
        test_set = Covid_loader_pt2(
                list_IDs = vl_indxs, 
                path = vl_path, 
                transform=test_transforms
        )
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.ts_bz, shuffle=False, num_workers=8, pin_memory=False)
        validate_loader = torch.utils.data.DataLoader(test_set, batch_size=args.ts_bz, num_workers=8, pin_memory=False)        
        
        start_time = time.time()
    
        total_correct_tr = 0
        total_correct_val = 0
           
        label_f1tr = []
        pred_f1tr = []
         
        for batch in tqdm(train_loader):
             images1, labels = batch
             images1 = images1.squeeze(dim=1).float().to(device)
             labels = labels.long().to(device)
    
             model.eval()
             with torch.no_grad():
                 preds = model(images1)
    
             total_correct_tr += get_num_correct(preds, labels)
    
             label_f1tr.extend(labels.cpu().numpy().tolist())
             pred_f1tr.extend(preds.argmax(dim=1).tolist())      
         
             del images1
             del labels
         
        label_f1vl = []
        pred_f1vl = []
        
    
        for batch in tqdm(validate_loader):
             images1, labels = batch
             images1 = images1.squeeze(dim=1).float().to(device)
    
             labels = labels.long().to(device)
    
             model.eval()
             with torch.no_grad():
                 preds = model(images1)
    
             total_correct_val += get_num_correct(preds, labels)
                 
             label_f1vl.extend(labels.cpu().numpy().tolist())
             pred_f1vl.extend(preds.argmax(dim=1).tolist())
             
             del images1
             del labels
             
        print('AC_ts1: ', total_correct_tr/len(train_set))    
        print('MaF1_ts1: ', f1_score(label_f1tr, pred_f1tr, average='macro'))     
        print('CM_ts1: ')
        print(confusion_matrix(label_f1tr, pred_f1tr))  
        
        with open(Resultstx, "a") as f:
    
            print('_' * 10, file=f) 
            print('Test1 Results', file=f) 
            print('Accuracy', accuracy_score(label_f1tr, pred_f1tr), file=f)
            print('F1-score', f1_score(label_f1tr, pred_f1tr, average='macro'), file=f)
            print(confusion_matrix(label_f1tr, pred_f1tr), file=f)              
             
             
    
        print('AC_ts2: ', total_correct_val/len(test_set))    
        print('MaF1_ts2: ', f1_score(label_f1vl, pred_f1vl, average='macro'))     
        print('CM_ts2: ')
        print(confusion_matrix(label_f1vl, pred_f1vl))  
        
        with open(Resultstx, "a") as f:
    
            print('_' * 10, file=f) 
            print('Test2 Results', file=f) 
            print('Accuracy', accuracy_score(label_f1vl, pred_f1vl), file=f)
            print('F1-score', f1_score(label_f1vl, pred_f1vl, average='macro'), file=f)
            print(confusion_matrix(label_f1vl, pred_f1vl), file=f)     
    
    

    
    
    















    
    
        # Augmentation test         
        """ Val"""

        ###########################
        
        vl_path = 'E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Val'
        vl_labels= np.load('E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Val/labels.npy')

        vl_indxs= list(range(len(vl_labels)))
        test_set = Covid_loader_pt3(
                list_IDs = vl_indxs, 
                path = vl_path, 
                transform=train_transforms
        )           
        
        
        validate_loader = torch.utils.data.DataLoader(test_set, batch_size=1)         
        
        label_f1vl = []
        pred_f1vl = [] 
        
        validation_loss = 0
        total_correct_val = 0        
        
        for batch in tqdm(validate_loader):
            images1, labels = batch
            images1 = images1.squeeze(dim=1).float().to(device)
    
            labels = labels.long().to(device)
    
            model.eval()
            with torch.no_grad():
                preds = model(images1.squeeze(dim=0))
                
            preds = torch.mean(preds, dim=0).unsqueeze(dim =0)
            total_correct_val += get_num_correct(preds, labels)        
    
            label_f1vl.extend(labels.cpu().numpy().tolist())
            pred_f1vl.extend(preds.argmax(dim=1).tolist())
            
            del images1
            del labels
    
        print(accuracy_score(label_f1vl, pred_f1vl))
        print(f1_score(label_f1vl, pred_f1vl, average='macro'))        
        print(confusion_matrix(label_f1vl, pred_f1vl))
        
        with open(Resultstx, "a") as f:
            print('-' * 10, file=f) 
            print('-' * 10, file=f) 
            print('-' * 10, file=f) 
            print('Test Augmentation Results' , file=f) 
            print('Val Results', file=f)         
            print('Accuracy', accuracy_score(label_f1vl, pred_f1vl), file=f)
            print('F1-score', f1_score(label_f1vl, pred_f1vl, average='macro'), file=f)
            print(confusion_matrix(label_f1vl, pred_f1vl), file=f)         
    
    
    
        """  Test 1, Test 2"""  
        tr_path = 'E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Test1'
        vl_path = 'E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Test2'

        tr_labels = np.load('E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Test1/labels.npy')
        vl_labels= np.load('E:/Fares/4th Covid-19 challenge/CH1/NumpysSevSeg/Test2/labels.npy')
        
        tr_indxs= list(range(len(tr_labels)))
        train_set = Covid_loader_pt3(
                list_IDs = tr_indxs, 
                path = tr_path, 
                transform=train_transforms
        )
        
        vl_indxs= list(range(len(vl_labels)))
        test_set = Covid_loader_pt3(
                list_IDs = vl_indxs, 
                path = vl_path, 
                transform=train_transforms
        )
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
        validate_loader = torch.utils.data.DataLoader(test_set, batch_size=1)        
        
        total_correct_val = 0
           
        label_f1vl = []
        pred_f1vl = []
        
        # Test1
        for batch in tqdm(train_loader):
            images1, labels = batch
            images1 = images1.squeeze(dim=1).float().to(device)
    
            labels = labels.long().to(device)
    
            model.eval()
            with torch.no_grad():
                preds = model(images1.squeeze(dim=0))
                
            preds = torch.mean(preds, dim=0).unsqueeze(dim =0)
    
            total_correct_val += get_num_correct(preds, labels)
                
            label_f1vl.extend(labels.cpu().numpy().tolist())
            pred_f1vl.extend(preds.argmax(dim=1).tolist())
            
            del images1
            del labels
    
        print('AC_ts1: ', total_correct_val/len(train_set))    
        print('MaF1_ts1: ', f1_score(label_f1vl, pred_f1vl, average='macro')) 
        print('CM_ts1: ')
        print(confusion_matrix(label_f1vl, pred_f1vl))  
        
        with open(Resultstx, "a") as f:
            
            print('_' * 10, file=f) 
            print('Test1 Results', file=f) 
            print('Accuracy', accuracy_score(label_f1vl, pred_f1vl), file=f)
            print('F1-score', f1_score(label_f1vl, pred_f1vl, average='macro'), file=f)
            print(confusion_matrix(label_f1vl, pred_f1vl), file=f)




        # test2    
        total_correct_val = 0
           
        label_f1vl = []
        pred_f1vl = []
        
    
        for batch in tqdm(validate_loader):
            images1, labels = batch
            images1 = images1.squeeze(dim=1).float().to(device)
    
            labels = labels.long().to(device)
    
            model.eval()
            with torch.no_grad():
                preds = model(images1.squeeze(dim=0))
                
            preds = torch.mean(preds, dim=0).unsqueeze(dim =0)
    
            total_correct_val += get_num_correct(preds, labels)
                
            label_f1vl.extend(labels.cpu().numpy().tolist())
            pred_f1vl.extend(preds.argmax(dim=1).tolist())
            
            del images1
            del labels
    
        print('AC_ts2: ', total_correct_val/len(test_set))    
        print('MaF1_ts2: ', f1_score(label_f1vl, pred_f1vl, average='macro')) 
        print('CM_ts2: ')
        print(confusion_matrix(label_f1vl, pred_f1vl))  
        
        with open(Resultstx, "a") as f:
            
            print('_' * 10, file=f) 
            print('Test2 Results', file=f) 
            print('Accuracy', accuracy_score(label_f1vl, pred_f1vl), file=f)
            print('F1-score', f1_score(label_f1vl, pred_f1vl, average='macro'), file=f)
            print(confusion_matrix(label_f1vl, pred_f1vl), file=f)
            
    
    
if __name__ == '__main__':
    train()            
            
            
            
            
            
            
            
            
            