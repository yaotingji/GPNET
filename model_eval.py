#eval model
import torch
import torch.utils.data as Data
import numpy as np
from torch import nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_score, recall_score, matthews_corrcoef, log_loss
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
torch.manual_seed(42)


class CoordinatetoPredict(nn.Module):
    def __init__(self):
        super(CoordinatetoPredict, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(4, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1), nn.BatchNorm1d(128), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1), nn.BatchNorm1d(1024), nn.ReLU())
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

coordinate = np.load('./testset.npz')['arr'].astype(np.float32)
print(coordinate.shape)
label = np.load('./testset_label.npz')['arr'].astype(np.float32)
test_coordinate = torch.from_numpy(coordinate)
test_coordinate = test_coordinate.permute(0, 2, 1)
test_label = torch.Tensor(label)

model=CoordinatetoPredict()
model.load_state_dict(torch.load('./trained_model_GPNET/train-038.pth'))
model.eval()
torch.no_grad()
outputs=model(test_coordinate)
test_label=test_label.float()
out_numpy=outputs.detach().numpy()
batchlabel_numpy_test=test_label.detach().numpy()
print("AUC_score:", roc_auc_score(batchlabel_numpy_test,out_numpy))
print("AUPR_score:", average_precision_score(batchlabel_numpy_test,out_numpy))
threshold = 0.148331 # the threshold of GPNET, and the threshold of fromscratch_model is 0.461227
test_pred = torch.where(outputs>threshold, torch.ones_like(outputs), torch.zeros_like(outputs))
print("accuracy_score:", accuracy_score(batchlabel_numpy_test,test_pred))
print("precision_score:", precision_score(batchlabel_numpy_test,test_pred))
print("recall_score:", recall_score(batchlabel_numpy_test,test_pred))
print("f1_score:", f1_score(batchlabel_numpy_test,test_pred))
print("MCC:", matthews_corrcoef(batchlabel_numpy_test,test_pred))
