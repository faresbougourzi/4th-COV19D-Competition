#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:33:55 2022

@author: bougourzi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 21:16:42 2021

@author: bougourzi
"""

from torch.utils.data import Dataset
import os
import os.path
import numpy as np
import torch
import cv2

import matplotlib.pyplot as plt


########################################## 


##########################################   
class Data_loader2(Dataset):
    def __init__(self, root, train, transform=None):

        self.train = train  # training set or test set
        self.data = np.load(os.path.join(root, train + "_Slice.npy"))
        self.y = np.load(os.path.join(root, train + "_Lung.npy"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, y = self.data[index], self.y[index]       

        # img1 = np.array(img)
        # y = np.array(y2)
        # # y2 = np.array(y2)
        # img1 = img1.astype(np.uint8) 
        # y = y.astype(np.uint8) 
        # y2 = y2.astype(np.uint8) 
        y[y > 0.0] = 1.0
              
        if self.transform is not None:
            augmentations = self.transform(image=img, mask=y)
            image = augmentations["image"]
            mask = augmentations["mask"]

            
        return   image, mask

    def __len__(self):
        return len(self.data)
##########################################   
    
import torch
from torch.utils import data   
##########################################   
    
class Covid_Seg_pt(data.Dataset):
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

        # Load data and get label
        # im_path = 
        img = cv2.imread(os.path.join(pth,'Slice', str(ID)+'.png'))        
        # img = (pth +'/'+ str(ID) + '.pt')
        y = cv2.imread(os.path.join(pth,'Inf', str(ID)+'.png'), cv2.IMREAD_GRAYSCALE) 
        # print(y.shape)
        # img, y, y2 = torch.load(pth +'/'+ str(ID) + '.pt'), cmap='gray'

        # img1 = np.array(img)
        # y = np.array(y)
        # # y2 = np.array(y2)
        # img1 = img1.astype(np.uint8) 
        # y = y.astype(np.uint8) 
        # y2 = y2.astype(np.uint8) 
        y[y > 0.0] = 1.0
              
        if self.transform is not None:
            augmentations = self.transform(image=img, mask=y)
            image = augmentations["image"]
            mask = augmentations["mask"]

            
        return   image, mask        
        
        


########################################## 

# class Covid_loader_pt(data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, list_IDs, labels, path):
#         'Initialization'
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.path = path

#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)

#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ID = self.list_IDs[index]
#         pth = self.path

#         # Load data and get label
#         X = torch.load(pth +'\\'+ ID + '.pt')
# #        print(X.shape)
# #        print(ID)
#         Id = int(ID)
#         y = self.labels[Id]

#         return X, y


  
# class Covid_loader_pt2(data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, list_IDs, path, transform=None):
#         'Initialization'
#         self.list_IDs = list_IDs
#         self.path = path
#         self.transform = transform

#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)

#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ID = self.list_IDs[index]
#         pth = self.path

#         # Load data and get label
#         # X, y1, y2, y3, y4  = torch.load(pth +'\\'+ ID + '.pt') 
#         X, y1, y2, y3, y4, y5  = torch.load( os.path.join(pth, ID + '.pt'))
#         # print(X.shape)
#         # print(ID)

        
#         if self.transform is not None:
#             X = self.transform(X)

#         return X, y1, y2, y3, y4, y5

    
    
    