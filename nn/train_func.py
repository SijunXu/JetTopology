import torch
import time
import os 
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn

def train_model(
    model, 
    dataloaders, 
    criterion, 
    optimizer, 
    num_epochs, 
    device, 
    idx, 
    folder2save, 
    name2save,
    verbose=False,
    hist_name=None
):
    r'''
    train a model with `idx`th in total 5 folds, save the model with best AUC while evaluating.
    name of save models are `folder2save + '/' + name2save + '_run' + str(idx) + '.pt'`
    save the training history if `hist_name`
    '''

    step_print = 5
    since = time.time()
    history = {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[]}
    best_val_loss = 999
    best_AUC = 0.5
    
    model = model.to(device)

    for epoch in range(num_epochs):
        if verbose:
            if epoch%step_print == 0:
                print('Epoch {0}/{1}'.format(epoch, num_epochs - 1))

        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        for i, data in enumerate(dataloaders['train']):

            for i_d in range(len(data)):
                data[i_d] = data[i_d].to(device).float()

            target = data[-1]
            optimizer.zero_grad()
            if len(data) == 2:
                outputs = model(data[0])
            elif len(data) == 5:
                outputs = model(data[0], data[1], data[2], data[3])

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * target.size(0)
            running_corrects += torch.sum((outputs+.5).int().t() == target.data.int().t())
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_corrects = running_corrects.double() / len(dataloaders['train'].dataset)
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_corrects)

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0.0

        for i, data in enumerate(dataloaders['val']):
            for i_d in range(len(data)):
                data[i_d] = data[i_d].to(device).float()

            target = data[-1]
            if len(data) == 2:
                outputs = model(data[0])
            elif len(data) == 5:
                outputs = model(data[0], data[1], data[2], data[3])            
            if i==0:
                val_out = outputs.detach().cpu().numpy()
                val_tar = target.detach().cpu().numpy()
            else:
                val_out = np.concatenate((val_out, outputs.detach().cpu().numpy()))
                val_tar = np.concatenate((val_tar, target.detach().cpu().numpy()))                
                
            loss = criterion(outputs, target)
            val_running_loss += loss.item() * target.size(0)
            val_running_corrects += torch.sum((outputs+.5).int().t() == target.data.int().t())
        val_epoch_loss = val_running_loss / len(dataloaders['val'].dataset)
        val_epoch_corrects = val_running_corrects.double() / len(dataloaders['val'].dataset)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_corrects)
        
        val_epoch_AUC = roc_auc_score(val_tar, val_out)
        if val_epoch_AUC >= best_AUC:
            best_AUC = val_epoch_AUC 
            f2save = os.path.join(folder2save, name2save + '_run' + str(idx) + '.pt')
            torch.save(model.state_dict(), f2save) 
            best_epoch = epoch
        
        if verbose:
            if epoch%step_print == 0:
                print('Epoch Loss: {0:.5f}, Acc: {1:.5f}, Val Loss: {2:.5f}, Val Acc: {3:.5f}'
                    .format(epoch_loss, epoch_corrects, val_epoch_loss, val_epoch_corrects))
                print('-' * 10)
    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('best AUC: {0:.3f}, at epoch {1:d}'.format(best_AUC, best_epoch))
    
    ## save `histpry`
    if hist_name:
        with open(hist_name, 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model, history


from . import JetData, fcn_net, PersNet

class Trainer:
    r'''
    train data in 5 folds 
    '''
    def __init__(self, name2load, layers, indim=50, verbose=False, loader_params=None, train_params=None, use_PersNet=False):
        r'''
        `train_params`: a dict with key `'folder2save', 'name2save', 'num_epochs', 'device', 'hist_name'`

        `loader_params`: a dict with key `'batch_size', 'shuffle', 'pin_memory', num_workers`
        '''
        super().__init__()
        ## train data to be loaded 
        self.name2load = name2load
        
        ## if use_PersNet layers is a dict with key b0_phi_layers, b1_phi_layers, b0_rho_layers, b1_rho_layers, fc_layers
        ## else the hidden layers for fcn
        self.layers = layers
        
        self.verbose = verbose        
        self.indim = indim        
        self.loader_params = loader_params
        self.train_params = train_params

        self.use_PersNet = use_PersNet
    
    def _train_iter(self, idx):
        r'''
        train net at `idx` th fold
        '''
        if self.verbose:
            print('training model with idx = {:d}'.format(idx))

        if self.use_PersNet:
            net = PersNet(b0_dim=5, b1_dim=4, **self.layers)
        else:
            net = fcn_net(layers=self.layers, indim=self.indim, BN=True)

        JD = JetData(train=True, idx=idx, name2load=self.name2load, loader_params=self.loader_params)
        if self.use_PersNet:
            train_loader, val_loader = JD.pers_data_loader()
        else:
            train_loader, val_loader = JD.data_loader()
        loaders = { 'train': train_loader, 'val': val_loader}
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
        net.parameters(),
            lr=0.005,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5
        )    

        net, history = train_model(
            model=net,
            dataloaders=loaders,
            criterion=criterion,
            optimizer=optimizer,            
            idx=idx,   
            verbose=self.verbose,
            **self.train_params
        )
        del net    


    def kfold_train(self, kfold=5):
        r'''
        train net for `kfold` folds
        '''
        foler2save = self.train_params['folder2save']
        if not os.path.exists(foler2save):
            os.makedirs(foler2save)
            
        for idx in range(kfold):
            self._train_iter(idx)    


class Evaluater(Trainer):
    '''
    evaluate model for 5 folds trainings
    '''
    def __init__(self):
        super().__init__()         
        self.device = self.loader_params['device']    
        self.loader_params['shuffle'] = False
        self.path2net = '/home/sijun/projects/TopologyAtCollider/JetTopology/saved_models/IRC_scan'
        if self.use_PersNet:
            self.path2net = os.path.join(self.path2net, 'PersNet')
        else:
            self.path2net = os.path.join(self.path2net, 'fcn')
        self.net_name = os.path.join(self.path2net, self.train_params['name2save'])

    def _evaluate_net(self, net, loader):
        
        net = net.to(self.device)  
        
        outputs, labels = [], []

        for i, data in enumerate(loader):
            for d_id in range(len(data)):
                data[i] = data[i].to(self.device)
            
            label = data[-1]
            
            if len(data) == 2:
                output = net(data[0])
            if len(data) == 5:
                output = net(data[0], data[1], data[2], data[3])

            labels.append(label.detach().cpu().numpy())
            outputs.append(output.detach().cpu().numpy())
        
        try:
            labels = np.concatenate(labels, axis=0)
            outputs = np.concatenate(outputs, axis=0)
        except:
            labels = np.concatenate(labels)
            outputs = np.concatenate(outputs)
        return outputs.reshape(-1), labels.reshape(-1)
    
    def _idx_eva(self, idx):
        if self.use_PersNet:
            net = PersNet(b0_dim=5, b1_dim=4, **self.layers)
        else:
            net = fcn_net(layers=self.layers, indim=self.indim, BN=True)

        weight2load = self.net_name + + '_run' + str(idx) + '.pt'
        net.load_state_dict(torch.load(weight2load))

        JD = JetData(train=False, idx=idx, name2load=self.name2load, loader_params=self.loader_params)
        loader  = JD.data_loader()
        y_pred, y_true = self._evaluate_net(net, loader)
        return y_pred, y_true

    def evaluate(self):
        '''
        return average output scores for 5 folds, labels and AUC score
        ''' 
        y_preds = []
        for idx in range(5):
            y_pred, y_true = self._idx_eva(idx)
            y_preds.append(y_preds)
        y_preds = np.array(y_preds).mean(axis=0)
        AUC = roc_auc_score(y_true, y_preds)
        return y_preds, y_true, AUC