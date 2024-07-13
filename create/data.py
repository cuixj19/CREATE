#!/usr/bin/env python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings("ignore")

LENGTH = 1000

def load_data(
        data_path, 
        num_class=5, 
        multi=['seq','open','hichip'], 
        test_aug=1, 
        train_aug=[1]*5, 
        stride=10, 
        batch_size=1024, 
        log=None, 
    ):
    """
    Load dataset with preprocessing
    
    Parameters
    ----------
    data_path
        A path of DNA sequence (chromatin accessibility and chromatin interaction data).
    num_class
        The number of types of CREs. Default: 1.
    multi
        Multimodal data used as input in the model. Default: ['seq','open','loop'].
    test_aug
        Augmentation for CREs during testing process. Default: 5.
    train_aug
        Augmentation for each kind of CREs during training process. Default: [1,1,1,1,1].
    stride
        Window stride for data augmentation. Default: 10.
    batch_size
        Number of samples per batch to load. Default: 1024.
    log
        If log, record each operation in the log file. Default: None.
    
    Returns
    -------
    train_loader
        An iterable over the given dataset for training.
    valid_loader
        An iterable over the given dataset for validating.
    test_loader
        An iterable over the given dataset for testing.
    valid_label
        CRE labels for validating.
    test_label
        CRE labels for testing.
    """

    if log: log.info('Use data: {}'.format('+'.join(multi)))
    
    train_labels = np.load(data_path+'train_labels.npy')
    valid_labels = np.load(data_path+'valid_labels.npy')
    test_labels = np.load(data_path+'test_labels.npy')
    if log: log.info('No.train={}, No.valid={}, No.test={}'.format(train_labels.shape[0], valid_labels.shape[0], test_labels.shape[0]))
    
    train_data, valid_data, test_data = [], [], []
    if 'seq' in multi:
        train_data.append(np.load(data_path+'train_seqs.npy'))
        valid_data.append(np.load(data_path+'valid_seqs.npy'))
        test_data.append(np.load(data_path+'test_seqs.npy'))
    
    if 'open' in multi:
        train_data.append(np.load(data_path+'train_opens.npy'))
        valid_data.append(np.load(data_path+'valid_opens.npy'))
        test_data.append(np.load(data_path+'test_opens.npy'))
    
    if 'loop' in multi:
        train_data.append(np.load(data_path+'train_loops.npy'))
        valid_data.append(np.load(data_path+'valid_loops.npy'))
        test_data.append(np.load(data_path+'test_loops.npy'))
    
    train_data = np.concatenate(train_data, axis=1)
    valid_data = np.concatenate(valid_data, axis=1)
    test_data = np.concatenate(test_data, axis=1)
    if log: log.info('The shape of input data: {}'.format(train_data.shape[1:]))
    
    data0, data_id, label = data_aug(train_data[train_labels==0], train_data[train_labels==1], train_data[train_labels==2], train_data[train_labels==3], train_data[train_labels==4], augs=train_aug, stride=stride, multi=multi)
    train_set = TensorDataset(torch.Tensor(data0), torch.Tensor(label), torch.Tensor(data_id))
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True, num_workers=4)
    
    data0, data_id, label = data_aug(valid_data[valid_labels==0], valid_data[valid_labels==1], valid_data[valid_labels==2], valid_data[valid_labels==3], valid_data[valid_labels==4], augs=[test_aug]*num_class, stride=stride, multi=multi)
    valid_set = TensorDataset(torch.Tensor(data0), torch.Tensor(label), torch.Tensor(data_id))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    data0, data_id, label = data_aug(test_data[test_labels==0], test_data[test_labels==1], test_data[test_labels==2], test_data[test_labels==3], test_data[test_labels==4], augs=[test_aug]*num_class, stride=stride, multi=multi)
    test_set = TensorDataset(torch.Tensor(data0), torch.Tensor(label), torch.Tensor(data_id))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    if log: log.info('Data loading completed!')
    
    return train_loader, valid_loader, test_loader, valid_labels, test_labels

def data_aug(s_all, e_all, p_all, i_all, b_all, augs=[5]*5, stride=10, multi=['seq','open','loop']):
    s_alls, s_num = load_all(s_all, aug=augs[0], stride=stride, multi=multi)
    e_alls, e_num = load_all(e_all, aug=augs[1], stride=stride, multi=multi)
    p_alls, p_num = load_all(p_all, aug=augs[2], stride=stride, multi=multi)
    i_alls, i_num = load_all(i_all, aug=augs[3], stride=stride, multi=multi)
    b_alls, b_num = load_all(b_all, aug=augs[4], stride=stride, multi=multi)
    
    data0 = []
    data_id = []
    num_s, num_e, num_p, num_i, num_b = 0, 0, 0, 0, 0
    for i in range(len(s_alls)):
        num_s += len(s_alls[i])
        data0.extend(s_alls[i])
        data_id.extend([i] * len(s_alls[i]))
    for i in range(len(e_alls)):
        num_e += len(e_alls[i])
        data0.extend(e_alls[i])
        data_id.extend([i] * len(e_alls[i]))
    for i in range(len(p_alls)):
        num_p += len(p_alls[i])
        data0.extend(p_alls[i])
        data_id.extend([i] * len(p_alls[i]))
    for i in range(len(i_alls)):
        num_i += len(i_alls[i])
        data0.extend(i_alls[i])
        data_id.extend([i] * len(i_alls[i]))
    for i in range(len(b_alls)):
        num_b += len(b_alls[i])
        data0.extend(b_alls[i])
        data_id.extend([i] * len(b_alls[i]))
    data_id = np.array(data_id)
    label = np.array([0] * num_s + [1] * num_e + [2] * num_p + [3] * num_i + [4] * num_b)
    
    return data0, data_id, label

def load_all(mats, aug=5, stride=10, multi=['seq','open','loop']):
    alls = []
    num = 0
    for i in range(len(mats)):
        s = mats[i]
        ss = split(s, aug=aug, stride=stride, multi=multi)
        num += len(ss)
        alls.append(ss)
    return alls, num

def split(s, aug=5, stride=10, multi=['seq','open','loop']):
    if 'seq' in multi:
        ids = [3,2,1,0] + [4+x for x in range(len(multi)-1)]
    else:
        ids = list(np.arange(len(multi)))
    w2 = LENGTH // 2
    lens = s.shape[1]
    ss = []
    if aug % 2 == 0:
        mid1 = int(lens/2)-int(stride/2)
        mid2 = int(lens/2)+int(stride/2)
        for j in range(aug//2):
            ss.append(s[:,mid1-j*stride-w2:mid1-j*stride+w2])
            ss.append(s[:,mid2+j*stride-w2:mid2+j*stride+w2])
            ss.append(s[:,mid1-j*stride-w2:mid1-j*stride+w2][ids,::-1])
            ss.append(s[:,mid2+j*stride-w2:mid2+j*stride+w2][ids,::-1])
    else:
        mid = int(lens/2)
        ss.append(s[:,mid-w2:mid+w2])
        ss.append(s[:,mid-w2:mid+w2][ids,::-1])
        for j in range(1, 1+(aug-1)//2):
            ss.append(s[:,mid-j*stride-w2:mid-j*stride+w2])
            ss.append(s[:,mid+j*stride-w2:mid+j*stride+w2])
            ss.append(s[:,mid-j*stride-w2:mid-j*stride+w2][ids,::-1])
            ss.append(s[:,mid+j*stride-w2:mid+j*stride+w2][ids,::-1])
    return ss

