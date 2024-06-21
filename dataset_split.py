import torch
import torch.utils.data as Data
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
torch.manual_seed(42)
import pandas as pd

	
name_collect=[]
label_collect=[]
with open('./GPERdata.txt',encoding='utf-8') as f: # your dataset file
    for line in f:
        line.rstrip()
        item=line.split()
        name=item[0]
        label=item[1]
        name_collect.append(name)
        label_collect.append(int(label))
        
sss=StratifiedShuffleSplit(n_splits=1,train_size=0.8,test_size=0.2,random_state=42)

for train_index, validation_index in sss.split(name_collect,label_collect):
    train_name = np.array(name_collect)[train_index]
    train_label = np.array(label_collect)[train_index]
    validation_name, validation_label= np.array(name_collect)[validation_index],np.array(label_collect)[validation_index]
    
    train_name=pd.DataFrame(train_name)
    train_name.to_csv('./train_name.csv')
    
    sss_validation=StratifiedShuffleSplit(n_splits=1,train_size=0.5,test_size=0.5,random_state=42)
    for internal_index, external_index in sss_validation.split(validation_name,validation_label):
        internal_name = np.array(validation_name)[internal_index]
        internal_label = np.array(validation_label)[internal_index]
        external_name, external_label= np.array(validation_name)[external_index],np.array(validation_label)[external_index]    
    
        internal_name=pd.DataFrame(internal_name)
        internal_name.to_csv('./validation_name.csv')
        external_name=pd.DataFrame(external_name)
        external_name.to_csv('./test_name.csv')