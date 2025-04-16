#!/usr/bin/env python
import os
import numpy as np
import torch

from .data import load_data
from .model.create_train import CREATE_train
from .model.layer import *
from .logger import create_logger

from sklearn import preprocessing
one_hot = preprocessing.OneHotEncoder(sparse=False)

import warnings
warnings.filterwarnings("ignore")

def CREATE(
        data_path, 
        num_class=5, 
        multi=['seq','open','loop'], 
        test_aug=1, 
        train_aug=[1]*5, 
        stride=10, 
        batch_size=1024, 
        enc_dims=[512, 384, 128], 
        dec_dims=[200, 200], 
        embed_dim=128, 
        n_embed=200, 
        split=16, 
        ema=True, 
        e_loss_weight=0.25, 
        mu=0.01, 
        open_loss_weight=0.01, 
        loop_loss_weight=0.1, 
        lr=5e-5, 
        max_epoch=300, 
        pre_epoch=50, 
        seed=0, 
        gpu=0, 
        outdir='./output/', 
    ):
    """
    Cis-Regulatory Elements identificAtion via discreTe Embedding
    
    Parameters
    ----------
    data_path
        A path of DNA sequence (chromatin accessibility and chromatin interaction data).
    num_class
        The number of types of CREs. Default: 5.
    multi
        Multimodal data used as input in the model. Default: ['seq','open','loop'].
    test_aug
        Augmentation for CREs during testing process. Default: 1.
    train_aug
        Augmentation for each kind of CREs during training process. Default: [1,1,1,1,1].
    stride
        Window stride for data augmentation. Default: 10.
    batch_size
        Number of samples per batch to load. Default: 1024.
    enc_dims
        The number of nodes in the layers of encoder. Default: [512, 384, 128].
    dec_dims
        The number of nodes in the layers of encoder. Default: [200, 200].
    embed_dim
        The dimension of latent embeddings. Default: 128.
    n_embed
        The size of codebook. Default: 200.
    split
        The number of split quantizations. Default: 16.
    ema
        If True, adopt the exponential moving average (EMA) to update the codebook instead of the codebook loss. Default: True.
    e_loss_weight
        The weight of encoder loss designed for encoder. Default: 0.25.
    mu
        The update ratio of codebook when ema is True. Default: 0.01.
    open_loss_weight
        The weight of reconstruction open loss. Default: 0.01.
    loop_loss_weight
        The weight of reconstruction loop loss. Default: 0.1.
    lr
        Learning rate. Default: 5e-5.
    max_epoch
        Max epochs for training. Default: 300.
    pre_epoch
        Pre-epochs for training before incorporating the classification loss. Default: 50.
    seed
        Random seed for torch and numpy. Default: 0.
    gpu
        Index of GPU to use if GPU is available. Default: 0.
    outdir
        Output directory. Default: './output/'.
    """
    
    os.makedirs(outdir+'/checkpoint', exist_ok=True)
    log = create_logger('', fh=outdir+'/log.txt')

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device='cuda'
        torch.cuda.set_device(gpu)
        log.info('Using GPU: {}'.format(gpu))
    else:
        device='cpu'
        log.info('Using CPU')
    
    train_loader, valid_loader, test_loader, valid_label, test_label = load_data(
        data_path, 
        num_class=num_class, 
        multi=multi, 
        test_aug=test_aug, 
        train_aug=train_aug, 
        stride=stride, 
        batch_size=batch_size, 
        log=log
    )
    
    channel1, channel2, channel3 = enc_dims
    channel4, channel5 = dec_dims
    clf = create(num_class=num_class, multi=multi, channel1=channel1, channel2=channel2, channel3=channel3, channel4=channel4, channel5=channel5, embed_dim=embed_dim, n_embed=n_embed, split=split, ema=ema, e_loss_weight=e_loss_weight, mu=mu)
#     log.info('model\n'+clf.__repr__())
    log.info('Model training...')

    CREATE_train(clf, 
                 train_loader, 
                 valid_loader, 
                 test_loader, 
                 valid_label, 
                 test_label, 
                 lr=lr, 
                 max_epoch=max_epoch, 
                 pre_epoch=pre_epoch, 
                 multi=multi, 
                 aug=test_aug*2, 
                 cls=num_class, 
                 open_loss_weight=open_loss_weight, 
                 loop_loss_weight=loop_loss_weight, 
                 outdir=outdir,
                 device=device
                )


