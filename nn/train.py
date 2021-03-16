import torch
import time
import os 
import pickle5 as pickle

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
            outputs = model(*data[:-1])            
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
            outputs = model(*data[:-1])           
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


import JetTopology.nn as tn
class Trainer:    
    r'''
    train data in 5 folds 
    '''
    def __init__(
        self, 
        name2load, 
        layers, 
        indim=50, 
        graph_type='DT', 
        verbose=False,
        loader_params=None, 
        train_params=None, 
        use_PersNet=False,
        use_obs=None,
        obs_fname=None,
        pers_only=False
        ):
        r'''
        `train_params`: a dict with key `'folder2save', 'name2save', 'num_epochs', 'device', 'hist_name'`

        `loader_params`: a dict with key `'batch_size', 'shuffle', 'pin_memory', num_workers`
        '''
        super().__init__()
        ## train data to be loaded 
        self.name2load = name2load
        
        ## if use_PersNet layers is a dict with key b0_phi_layers, b1_phi_layers, b0_rho_layers, b1_rho_layers, fc_layers
        ## elif use fcn the hidden layers for fcn
        ## elif use net to topo + obs
        self.layers = layers
        
        self.verbose = verbose        
        self.indim = indim        
        self.loader_params = loader_params
        self.train_params = train_params
        self.graph_type = graph_type
        self.use_PersNet = use_PersNet
        self.use_obs = use_obs
        self.obs_fname = obs_fname
        self.pers_only = pers_only
    
    def _train_iter(self, idx):
        r'''
        train net at `idx` th fold
        '''
        if self.verbose:
            print('training model with idx = {:d}'.format(idx))

        ### get net for training
        if self.use_PersNet:
            if self.graph_type == 'DT':
                net = PersNet(b0_dim=5, b1_dim=4, **self.layers)
            elif self.graph_type == 'kNN':
                net = PersNetkNN(b0_dim=5, **self.layers)  
        else:
            if self.use_obs is None:
                net = fcn_net(layers=self.layers, indim=self.indim, BN=True)
            else:
                if self.use_obs == 'with_topo':
                    net = TopoObsNet(**self.layers, topo_indim=self.indim)
                else:
                    net = fcn_net(layers=self.layers, indim=6)        

        ### get data loaders
        JD = JetData(train=True, idx=idx, name2load=self.name2load, loader_params=self.loader_params)
        if self.use_PersNet:
            if self.graph_type == 'DT':
                train_loader, val_loader = JD.pers_data_loader()
            if self.graph_type == 'kNN':
                train_loader, val_loader = JD.kNN_pers_data_loader()
        else:
            if self.graph_type == 'DT':
                if self.use_obs is None:
                    train_loader, val_loader = JD.data_loader()
                else:
                    with_topo = ( self.use_obs == 'with_topo' )
                    train_loader, val_loader = JD.obs_data_loader(self.obs_fname, with_topo=with_topo, pers_only=self.pers_only)                                        
            elif self.graph_type == 'kNN':
                train_loader, val_loader = JD.kNN_data_loader()

        ### train
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