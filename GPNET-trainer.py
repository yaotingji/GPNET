#transfer learning from ER to GPER

import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
import sklearn
from sklearn import metrics
import sys

torch.manual_seed(42)

coordinate=np.load('./trainset.npz')['arr'].astype(np.float32)
print(coordinate.shape)
label=np.load('./trainset_label.npz')['arr'].astype(np.float32)
train_coordinate=torch.from_numpy(coordinate)
train_coordinate=train_coordinate.permute(0,2,1)
train_label=torch.Tensor(label) 
trainset=Data.TensorDataset(train_coordinate,train_label)
BATCH_SIZE_train=33
trainset_loader=Data.DataLoader(trainset,batch_size=BATCH_SIZE_train,drop_last=False)


coordinate=np.load('./validationset.npz')['arr'].astype(np.float32)
print(coordinate.shape)
label=np.load('./validationset_label.npz')['arr'].astype(np.float32)
validation_coordinate=torch.from_numpy(coordinate)
validation_coordinate=validation_coordinate.permute(0,2,1)
validation_label=torch.Tensor(label)

#freeze conv1-4
class CoordinatetoPredict(nn.Module):
    def __init__(self):
        super(CoordinatetoPredict,self).__init__()
        self.conv1=nn.Sequential(nn.Conv1d(4,64,kernel_size=1),nn.BatchNorm1d(64),nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv1d(64,64,kernel_size=1),nn.BatchNorm1d(64),nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv1d(64,128,kernel_size=1),nn.BatchNorm1d(128),nn.ReLU())
        self.conv4=nn.Sequential(nn.Conv1d(128,1024,kernel_size=1),nn.BatchNorm1d(1024),nn.ReLU())
        for p in self.parameters():
            p.requires_grad=False
        self.conv5=nn.Sequential(nn.MaxPool1d(kernel_size=4096))

        self.fc=nn.Sequential(nn.Linear(1024,256),nn.BatchNorm1d(256),nn.ReLU(),
                              nn.Linear(256,64),nn.BatchNorm1d(64),nn.ReLU(),
                              nn.Linear(64,8),nn.BatchNorm1d(8),nn.ReLU(),
                              nn.Linear(8,1),nn.Sigmoid())
                             
    def forward(self, x):
        x_conv1=self.conv1(x)
        x_conv2=self.conv2(x_conv1)
        x_conv3=self.conv3(x_conv2)
        x_conv4=self.conv4(x_conv3)
        x_conv5=self.conv5(x_conv4)
        x_conv5=x_conv5.view(-1,x_conv5.size(1))
        x_fc=self.fc(x_conv5)
        x_out=x_fc.view(x_fc.size(0))
        return x_out
        
model=CoordinatetoPredict()
model.load_state_dict(torch.load('./pretrained_model_ER/train-014.pth'))
optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),weight_decay=0.0309,lr=0.0158)
criterion=nn.BCELoss()

with open ('./transfer_model/result/trainres.txt',mode='w',encoding="utf-8") as h:
	print('Epoch\tLoss\tAUC',file=h)
with open ('./transfer_model/result/validationres.txt',mode='w',encoding="utf-8") as f:
	print('Epoch\tLoss\tAUC',file=f)
AUC_validation_max=0.5
consecutiveepoch_num=0
for epoch in range(60):
    loss_epoch=0
    label_epoch=[]
    out_epoch=[]
    model.train()
    for batch_data_train in trainset_loader:
        batch_coordinate_train, batch_label_train=batch_data_train
        print(batch_coordinate_train.shape)
        out=model(batch_coordinate_train)
        batch_label_train=batch_label_train.float()
        loss_train=criterion(out,batch_label_train)
        batch_loss_train=loss_train.item()
        loss_epoch=loss_epoch+batch_loss_train*batch_label_train.size(0)
        out_numpy=out.detach().numpy()
        batchlabel_numpy_train=batch_label_train.detach().numpy()
        out_epoch=np.concatenate((out_epoch,out_numpy),axis=0)
        label_epoch=np.concatenate((label_epoch,batchlabel_numpy_train),axis=0)
        AUC_train=metrics.roc_auc_score(label_epoch,out_epoch)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    if (epoch+1) %1 ==0:
        print('*'*10)
        print('epoch {}, Loss {}'.format((epoch+1),(loss_epoch/120)))	
        print('AUC of Training set: {}'.format(AUC_train))
        with open('./transfer_model/result/trainres.txt',mode='a',encoding="utf-8") as h:
            print(str(epoch+1)+'\t'+str(loss_epoch/120)+'\t'+str(AUC_train),file=h)
            

    model.eval()
    out=model(validation_coordinate)
    validation_label=validation_label.float()
    loss_validation=criterion(out,validation_label)
    loss_validation=loss_validation.item()
    out_numpy=out.detach().numpy()
    batchlabel_numpy_validation=validation_label.detach().numpy()
    AUC_validation=metrics.roc_auc_score(batchlabel_numpy_validation,out_numpy)
    print('*'*10)
    print('epoch {}, Loss {}'.format((epoch+1),(loss_validation)))
    print('AUC of Validation set: {}'.format(AUC_validation))
    if (epoch+1) % 1 ==0:
        with open('./transfer_model/result/validationres.txt',mode='a',encoding="utf-8") as f:
            print(str(epoch+1)+'\t'+str(loss_validation)+'\t'+str(AUC_validation),file=f)
    if (epoch+1) % 1 ==0:
        torch.save(model.state_dict(),'./transfer_model/ref/train-{:03d}.pth'.format(epoch+1))
    
    if AUC_validation > AUC_validation_max:
        AUC_validation_max=AUC_validation
        consecutiveepoch_num=0
        print('{} consecutive epoches without AUC increase'.format(consecutiveepoch_num))
    else: 
        consecutiveepoch_num+=1
        print('{} consecutive epoches without AUC increase'.format(consecutiveepoch_num))
    
    if consecutiveepoch_num>=15:
        sys.exit(0)
        
        
                  