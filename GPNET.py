#GPNET_prediction.py
import torch
import torch.utils.data as Data
import numpy as np
from torch import nn
import torch.nn.functional as F
import pandas as pd
torch.manual_seed(42)
import os,glob,re
import sys, getopt

def input_prediction(outputfile, threshold=True):
    coordinate = np.load('./GPER_pred.npz')['arr'].astype(np.float32)
    pred_coordinate = torch.from_numpy(coordinate)
    pred_coordinate = pred_coordinate.permute(0, 2, 1)

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

#model prediction
    model=CoordinatetoPredict()
    model.load_state_dict(torch.load('./trained_model_GPNET/train-038.pth'))
    print('Loading model...')
    model.eval()
    torch.no_grad()
    outputs=model(pred_coordinate)
    out_numpy=outputs.detach().numpy()

    print('Writing outputfile...')
    if threshold == True:
        pred = torch.where(outputs>=0.148331, torch.ones_like(outputs), torch.zeros_like(outputs))
        preds = pred.detach().numpy()
        df = pd.DataFrame(data=preds, columns=['activity_classification'])
        df.to_csv(outputfile)
    else:
        df = pd.DataFrame(data=out_numpy, columns=['activity_score'])
        df.to_csv(outputfile)

    print('Prediction done！')

def main(argv):
    outputfile = ''
    threshold = True
    try:
        opts, args = getopt.getopt(argv, "o:t", ["outputfile=", "threshold="])
        if opts == []:
            print(usage.__doc__)
    except getopt.GetoptError as err:
        print(err)
        print(usage.__doc__)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-o", "--outputfile"):
            outputfile = arg
        elif opt in ("-t", "--threshold"):
            threshold = arg
    if outputfile == '':
        print('Warning：the outputfile is empty，please set the outputfile！')
    else:
        print('Set outputfile：', outputfile)
        print('Set threshold:', threshold)
        input_prediction(outputfile, threshold)


if __name__ == '__main__':
    main(sys.argv[1:])




