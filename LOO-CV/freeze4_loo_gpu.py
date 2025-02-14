import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import sklearn
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
import sys
import json
import os

class CoordinatetoPredict(nn.Module):
    def __init__(self):
        super(CoordinatetoPredict, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(4, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1), nn.BatchNorm1d(128), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1), nn.BatchNorm1d(1024), nn.ReLU())
        for p in self.parameters():
            p.requires_grad = False
        self.conv5 = nn.Sequential(nn.MaxPool1d(kernel_size=4096))

        self.fc = nn.Sequential(nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                nn.Linear(64, 8), nn.BatchNorm1d(8), nn.ReLU(),
                                nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv5 = x_conv5.view(-1, x_conv5.size(1))
        x_fc = self.fc(x_conv5)
        x_out = x_fc.view(x_fc.size(0))
        return x_out


def train_evaluate_loo(batchsize, weightdecay, lr_init):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coordinate = np.load('./LOOset.npz')['arr'].astype(np.float32)
    label = np.load('./LOOset_label.npz')['arr'].astype(np.float32)
    train_coordinate = torch.from_numpy(coordinate).permute(0, 2, 1).to(device)
    train_label = torch.Tensor(label).to(device)

    loo = LeaveOneOut()
    AUC_validation_max = 0.5
    consecutiveepoch_num = 0
    epochs = 60

    model = CoordinatetoPredict().to(device)
    model.load_state_dict(torch.load('./pretrained_model_ER/train-{:03d}.pth'.format(14)))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=weightdecay, lr=lr_init)
    criterion = nn.BCELoss()

    with open("./freeze4_val/freeze4_3_val_results.txt", "w") as result_file:
        for epoch in range(epochs):
            model.train()
            all_val_labels = []
            all_val_outs = []

            for train_index, val_index in loo.split(train_coordinate):
                train_data, val_data = train_coordinate[train_index], train_coordinate[val_index]
                train_labels, val_labels = train_label[train_index], train_label[val_index]

                trainset = Data.TensorDataset(train_data, train_labels)
                trainset_loader = Data.DataLoader(trainset, batch_size=batchsize, drop_last=True)

                for batch_data_train in trainset_loader:
                    batch_coordinate_train, batch_label_train = batch_data_train
                    out = model(batch_coordinate_train)
                    batch_label_train = batch_label_train.float().to(device)
                    loss_train = criterion(out, batch_label_train)

                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    out_val = model(val_data)
                    val_labels = val_labels.float().to(device)
                    out_val_numpy = out_val.detach().cpu().numpy()
                    val_labels_numpy = val_labels.detach().cpu().numpy()

                    all_val_outs.append(out_val_numpy)
                    all_val_labels.append(val_labels_numpy)

            all_val_labels = np.concatenate(all_val_labels, axis=0)
            all_val_outs = np.concatenate(all_val_outs, axis=0)

            AUC_val = metrics.roc_auc_score(all_val_labels, all_val_outs)
            print('Epoch: {} - AUC: {}'.format(epoch + 1, AUC_val))

            torch.save(model.state_dict(), f"./freeze4_val/freeze4_3_saved_models/epoch_{epoch + 1}.pth")

            result_file.write(f"Epoch: {epoch + 1}, AUC: {AUC_val}\n")
            result_file.flush()

            if AUC_val > AUC_validation_max:
                AUC_validation_max = AUC_val
                consecutiveepoch_num = 0

            else:
                consecutiveepoch_num += 1
                print('{} consecutive epochs without AUC improvement'.format(consecutiveepoch_num))

            if consecutiveepoch_num >= 15:
                print('Early stopping triggered')
                break

    return AUC_validation_max

batchsize = 27
weightdecay = 0.0608561338000804
lr_init = 0.000219701261471005
train_evaluate_loo(batchsize, weightdecay, lr_init)



