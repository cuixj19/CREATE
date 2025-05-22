#!/usr/bin/env python
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn import metrics
from sklearn import preprocessing
one_hot = preprocessing.OneHotEncoder(sparse=False)

from .layer import *
from .utils import *


def CREATE_train(
        clf, 
        train_loader, 
        valid_loader, 
        test_loader, 
        valid_label, 
        test_label, 
        lr=5e-5, 
        max_epoch=300, 
        pre_epoch=50, 
        multi=['seq','open','loop'], 
        aug=2, 
        cls=5, 
        open_loss_weight=0.01, 
        loop_loss_weight=0.1, 
        outdir='./output/', 
        device='cuda'
    ):
    """
    Parameters
    ----------
    clf
        The CREATE model.
    train_loader
        Dataloader for training dataset.
    valid_loader
        Dataloader for validation dataset.
    test_loader
        Dataloader for testing dataset.
    valid_label
        CRE labels for validation dataset.
    test_label
        CRE labels for testing dataset.
    lr
        Learning rate. Default: 5e-5.
    max_epoch
        Max epochs for training. Default: 300.
    pre_epoch
        Pre-epochs for training before incorporating the classification loss. Default: 50.
    multi
        Multimodal data used as input in the model. Default: ['seq','open','loop'].
    aug
        Augmentation for CREs during testing process. Default: 2.
    cls
        The number of types of CREs. Default: 5.
    open_loss_weight
        The weight of reconstruction open loss. Default: 0.01.
    loop_loss_weight
        The weight of reconstruction loop loss. Default: 0.1.
    outdir
        Output directory. Default: './output/'.
    device
        'cuda' or 'cpu' for training. Default: 'cuda'.
    """
    clf = clf.to(device)
    
    valid_labels = one_hot.fit_transform(np.array(valid_label).reshape(len(valid_label), 1))
    test_labels = one_hot.fit_transform(np.array(test_label).reshape(len(test_label), 1))
    ids = {'seq':[],'open':[],'loop':[]}
    dicts = {'seq':4,'open':1,'loop':1}
    last = 0
    for omic in multi:
        ids[omic] = [last, last+dicts[omic]]
        last = last+dicts[omic]
    
    optimizer = optim.Adam(clf.parameters(), lr=lr)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-6)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=1e-8)
    clf_loss1 = nn.CrossEntropyLoss()
    clf_loss2 = FocalLoss()
    recon_loss1_ = nn.BCELoss()
    recon_loss2_ = nn.MSELoss()
    regloss = Regularization(clf, weight_decay1=1e-8, weight_decay2=5e-7)

    loss1_rate = [0.0] * pre_epoch + [1.0] * (max_epoch - pre_epoch)
    loss2_rate = [1.0] * pre_epoch + [0.5] * (max_epoch - pre_epoch)
    max_prc = 0.0
    max_prc_epoch = 0
    
    with tqdm(range(max_epoch), total=max_epoch, desc='Epochs') as tq:
        for epoch in tq:
            train_out, train_label, valid_out, test_out = [], [], [], []
            training_loss, training_clf, training_recon, training_latent, training_perplexity = [], [], [], [], []
            clf.train()
            for _, (b_x0, b_y, _) in enumerate(train_loader):
                train_label.extend(list(b_y))
                b_x0 = b_x0.to(device)
                output, x0, out, _, latent_loss, perplexity3 = clf(b_x0)
                train_out.extend(F.softmax(output, 1).cpu().detach().numpy())

                reg_loss = regloss(clf)
                clf_loss = clf_loss1(output, b_y.to(device).long()) + clf_loss2(x0[:,0], ((b_y == 0) * 1).to(device).float()) + reg_loss
                recon_loss = 0.0 * reg_loss
                if 'seq' in multi:
                    i1, i2 = ids['seq']
                    recon_loss += recon_loss1_(out[:, i1:i2], b_x0[:, i1:i2])
                if 'open' in multi:
                    i1, i2 = ids['open']
                    recon_loss += open_loss_weight * recon_loss2_(out[:, i1:i2], b_x0[:, i1:i2])
                if 'loop' in multi:
                    i1, i2 = ids['loop']
                    recon_loss += loop_loss_weight * recon_loss2_(out[:, i1:i2], b_x0[:, i1:i2])
                loss = loss1_rate[epoch] * (clf_loss + reg_loss) + loss2_rate[epoch] * (recon_loss + latent_loss)
                training_clf.append(torch.mean(clf_loss).item())
                training_recon.append(torch.mean(recon_loss).item())
                training_latent.append(torch.mean(latent_loss).item())
                training_perplexity.append(torch.mean(perplexity3).item())
                training_loss.append(torch.mean(loss).item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss = float(np.mean(training_loss))
            train_recon = float(np.mean(training_recon))
            train_latent = float(np.mean(training_latent))
            train_clf = float(np.mean(training_clf))
            train_perplexity = float(np.mean(training_perplexity))
            train_labels = one_hot.fit_transform(np.array(train_label).reshape(len(train_label), 1))
            train_score = np.array(train_out)
            train_pre = [list(x).index(max(x)) for x in train_score]
            
            epoch_info = 'recon=%.3f, latent=%.3f, perplexity=%.3f' % (train_recon, train_latent, train_perplexity) if epoch < pre_epoch else 'recon=%.3f, latent=%.3f, clf=%.3f, perplexity=%.3f' % (train_recon, train_latent, train_clf, train_perplexity)
            tq.set_postfix_str(epoch_info)

            clf.eval()
            with torch.no_grad():
                for _, (b_x0, b_y, _) in enumerate(valid_loader):
                    output, x0, _, _, _, _ = clf(b_x0.to(device))
                    valid_out.extend(F.softmax(output, 1).cpu().detach().numpy())
                valid_out = np.array(valid_out)
                valid_score = get_mean_score(valid_label, valid_out, aug=aug, cls=cls)
                valid_pre = [list(x).index(max(x)) for x in valid_score]
                valid_prc = metrics.average_precision_score(valid_labels, valid_score)

                for _, (b_x0, b_y, _) in enumerate(test_loader):
                    output, _, _, _, _, _ = clf(b_x0.to(device))
                    test_out.extend(F.softmax(output, 1).cpu().detach().numpy())
                test_out = np.array(test_out)
                test_score = get_mean_score(test_label, test_out, aug=aug, cls=cls)
                test_pre = [list(x).index(max(x)) for x in test_score]

            if epoch >= pre_epoch:
                print("Train accuracy: %.6f" % metrics.accuracy_score(train_label, train_pre))
                print("Train auROC: %.6f" % metrics.roc_auc_score(train_label, train_score, multi_class='ovr'))
                print("Train auPRC: %.6f" % metrics.average_precision_score(train_labels, train_score))
                print("Valid accuracy: %.6f" % metrics.accuracy_score(valid_label, valid_pre))
                print("Valid auROC: %.6f" % metrics.roc_auc_score(valid_label, valid_score, multi_class='ovr'))
                print("Valid auPRC: %.6f" % valid_prc)
                print("Test accuracy: %.6f" % metrics.accuracy_score(test_label, test_pre))
                print("Test auROC: %.6f" % metrics.roc_auc_score(test_label, test_score, multi_class='ovr'))
                print("Test auPRC: %.6f" % metrics.average_precision_score(test_labels, test_score))

            if epoch < pre_epoch:
                scheduler1.step(train_recon + train_latent)
            else:
                scheduler2.step(train_loss)

            if valid_prc > max_prc:
                max_prc = valid_prc
                max_prc_epoch = epoch
                np.save(outdir+'best_test_score.npy', test_score)
                if epoch >= pre_epoch - 1:
                    state = {'model': clf.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': (epoch + 1)}
                    torch.save(state, outdir+'checkpoint/best_model.pth')
            if epoch - max_prc_epoch >= 20 and epoch >= 200:
                print("Early stop!")
                break

        torch.cuda.empty_cache()

    
