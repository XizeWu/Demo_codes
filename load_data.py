from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import h5py
import scipy.io as sio
import os

torch.multiprocessing.set_sharing_strategy('file_system')

class CustomDataSet(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        return img, label, index

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count

def get_loader(batch_size, dset):

    if dset=='mir':
        path = '/data/s2019020823/ResNetFeat/MIR_ResNet/'
        train_set = h5py.File(path + 'mir_train.mat', 'r', libver='latest', swmr=True)
        train_L = np.array(train_set['L_tr'], dtype=np.float).T
        train_x = np.array(train_set['I_tr'], dtype=np.float).T
        train_set.close()

        query_set = h5py.File(path + 'mir_query.mat', 'r', libver='latest', swmr=True)
        query_L = np.array(query_set['L_te'], dtype=np.float).T
        query_x = np.array(query_set['I_te'], dtype=np.float).T
        query_set.close()

        db_set = h5py.File(path + 'mir_database.mat', 'r', libver='latest', swmr=True)
        database_L = np.array(db_set['L_db'], dtype=np.float).T
        database_x = np.array(db_set['I_db'], dtype=np.float).T
        db_set.close()
    elif dset=='nus':
        path = '/data/s2019020823/ResNetFeat/NUS_ResNet/'
        train_set = h5py.File(path + 'nus_train.mat', 'r', libver='latest', swmr=True)
        train_L = np.array(train_set['L_tr'], dtype=np.float).T
        train_x = np.array(train_set['I_tr'], dtype=np.float).T
        train_set.close()

        query_set = h5py.File(path + 'nus_query.mat', 'r', libver='latest', swmr=True)
        query_L = np.array(query_set['L_te'], dtype=np.float).T
        query_x = np.array(query_set['I_te'], dtype=np.float).T
        query_set.close()

        db_set = h5py.File(path + 'nus_database.mat', 'r', libver='latest', swmr=True)
        database_L = np.array(db_set['L_db'], dtype=np.float).T
        database_x = np.array(db_set['I_db'], dtype=np.float).T
        db_set.close()
    elif dset=='coco':
        path = '/data/s2019020823/ResNetFeat/COCO_ResNet/'
        train_set = h5py.File(path + 'coco_train.mat', 'r', libver='latest', swmr=True)
        train_L = np.array(train_set['L_tr'], dtype=np.float).T
        train_x = np.array(train_set['I_tr'], dtype=np.float).T
        train_set.close()

        query_set = h5py.File(path + 'coco_query.mat', 'r', libver='latest', swmr=True)
        query_L = np.array(query_set['L_te'], dtype=np.float).T
        query_x = np.array(query_set['I_te'], dtype=np.float).T
        query_set.close()

        db_set = h5py.File(path + 'coco_database.mat', 'r', libver='latest', swmr=True)
        database_L = np.array(db_set['L_db'], dtype=np.float).T
        database_x = np.array(db_set['I_db'], dtype=np.float).T
        db_set.close()
    else:
        print("Dataname:", dset, 'Error!')



    imgs = {'train': train_x, 'query': query_x, 'database': database_x}
    labels = {'train': train_L, 'query': query_L, 'database': database_L}

    dataset = {x: CustomDataSet(images=imgs[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}
    # shuffle = {'query': False, 'train': False, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in
                  ['query', 'train', 'database']}

    return dataloader
