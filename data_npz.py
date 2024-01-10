import numpy as np
import sys

coordinate_collect_train=[]
label_collect_train=[]
with open('./trainset.txt',encoding='utf-8') as f:
    for line in f:
        line.rstrip()
        item=line.split()
        name=item[0]
        label=item[1]
        coordinate=[]
        with open('./datasets/'+name+'sparse.txt',encoding='utf-8') as h:
            for hang in h:
                hang.rstrip()
                hangproper=hang.split()
                coordinate.append((float(hangproper[0]),float(hangproper[1]),float(hangproper[2]),float(hangproper[3])))
        coordinate_collect_train.append(coordinate)
        label_collect_train.append(int(label))

coordinate_collect_train=np.array(coordinate_collect_train)
np.savez_compressed('./trainset.npz', arr=coordinate_collect_train)
label_collect_train=np.array(label_collect_train)
np.savez_compressed('./trainset_label.npz', arr=label_collect_train)
# loading .npz file directly helps to save time




