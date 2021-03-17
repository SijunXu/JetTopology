import torch
from torch.utils import data
import numpy as np
import pickle5 as pickle


class TableDataset(data.Dataset):

    def __init__(self, X, target):
        self.X = torch.from_numpy(X).float()
        target = np.reshape(target, (-1, 1)).astype(np.float32)
        self.target = torch.from_numpy(target)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.target[index]        
        return x, y

    def __len__(self):
        return len(self.X)



class TableObsDataset(data.Dataset):

    def __init__(self, X, obs, target):
        self.X = torch.from_numpy(X).float()
        self.obs = torch.from_numpy(obs).float()
        target = np.reshape(target, (-1, 1)).astype(np.float32)
        self.target = torch.from_numpy(target)

    def __getitem__(self, index):
        x = self.X[index]
        obs = self.obs[index]
        y = self.target[index]        
        return x, obs, y

    def __len__(self):
        return len(self.X)


class PersDataset(data.Dataset):
    '''
    Dataset for PersNet
    '''
    def __init__(self, b0, b0_weight, b1, b1_weight, target):        
        
        shape = b0_weight.shape         
        b0_weight = b0_weight.reshape((shape[0], shape[1], 1))
    
        shape = b1_weight.shape         
        b1_weight = b1_weight.reshape((shape[0], shape[1], 1))
        target = np.reshape(target, (-1, 1)).astype(np.float32)

        self.b0, self.b0_weight, self.b1, self.b1_weight, self.target = map(torch.from_numpy, (b0, b0_weight, b1, b1_weight, target))

    def __getitem__(self, index):
        return self.b0[index], self.b0_weight[index], self.b1[index], self.b1_weight[index], self.target[index]

    def __len__(self):
        return len(self.target)


class PerskNNDataset(data.Dataset):
    '''
    Dataset for PersNet for kNN based graph
    '''
    def __init__(self, b0, b0_weight, target):        
        
        shape = b0_weight.shape         
        b0_weight = b0_weight.reshape((shape[0], shape[1], 1))
        target = np.reshape(target, (-1, 1)).astype(np.float32)

        self.b0, self.b0_weight, self.target = map(torch.from_numpy, (b0, b0_weight, target))

    def __getitem__(self, index):
        return self.b0[index], self.b0_weight[index], self.target[index]

    def __len__(self):
        return len(self.target)    


from sklearn.model_selection import train_test_split

class JetData:
    r'''
    convert a dic of {'b0', 'b1', 'y'} / npz file, to `DataLoader` object
    '''
    def __init__(
        self, 
        train=True, 
        idx=None, 
        name2load=None, 
        loader_params=None,
        use_random=False,
        ):        
        '''
        `loader_params`: a dict to contral `batch_size`, `shuffle`, `pin_memory`, `num_workers`
        '''
        if idx is not None:
            self.idx = idx  
        self.train = train        
        self.name2load = name2load         
        if loader_params:
            self.loader_params = loader_params
        else: ## default values
            self.loader_params = {
                    'batch_size': 128,
                    'shuffle': False,
                    'pin_memory': False,
                    'num_workers': 2
                }
        ### using sklearn train_test_split 
        self.use_random = use_random

    def _load_data(self): 
        with open(self.name2load, 'rb') as handle:
            x = pickle.load(handle)
            return x 
                
    def _train_val_split(self, train_data, kfold=5, idx=0):
        r'''
        input `train_data` as a list
        returns `train_data_tra` and `train_data_val` for 5 folds    
        '''
        if self.use_random:
            random_state = idx
            train_data_tra, train_data_val = train_test_split(train_data, random_state=random_state, shuffle=True, test_size=0.111111)
            return train_data_tra, train_data_val
        else:            
            nb_val = int( 20000/5 )        
            train_data_tra = []
            train_data_val = []
            for i in range(5):
                ## loop over 5 pt bins 
                full_idx = list(range(i*20000, (i+1)*20000))            
                val_idx = list(range(i*20000+idx*nb_val, i*20000+(idx+1)*nb_val))
                train_idx = list( set(full_idx)-set(val_idx) )
                train_idx += [100000 + evt_idx for evt_idx in train_idx]
                val_idx += [100000 + evt_idx for evt_idx in val_idx]
                train_data_tra += [ train_data[idx] for idx in train_idx ]
                train_data_val += [ train_data[idx] for idx in val_idx ]                    
            return train_data_tra, train_data_val    

    def _make_persnet_dataset(self, y, b0, b1=None, max_b0=None, max_b1=None):
        '''
        convert a list of arrays to array
        '''
        n_jets = len(b0)        
        if max_b0 is None:
            max_b0 = 0
            for p0 in b0:
                max_b0 = max(len(p0), max_b0)        
        if max_b1 is None:
            max_b1 = 0
            for p1 in b1:
                max_b1 = max(len(p1), max_b1)      
        b0_feat, b0_weight = np.zeros((n_jets, max_b0, 5)), np.zeros((n_jets, max_b0))
        for i, p0 in enumerate(b0):
            n_b0 = len(p0)
            if n_b0 == 1:
                b0_feat[i, 0] = list(p0[0])
            elif n_b0 > 1:
                b0_feat[i, :n_b0] = p0        
        ## weight = log(b) - log(d)
        b0_weight = b0_feat[:, :, 0] - b0_feat[:, :, 1]
        if b1 is not None:
            b1_feat, b1_weight = np.zeros((n_jets, max_b1, 4)), np.zeros((n_jets, max_b1))
            for i, p1 in enumerate(b1):
                n_b1 = len(p1)
                if n_b1 == 1:
                    b1_feat[i, 0] = list(p1[1])
                elif n_b1 > 1:
                    b1_feat[i, :n_b1] = p1        
            b1_weight = b1_feat[:, :, 1] - b1_feat[:, :, 0]             
            return b0_feat, b1_feat, b0_weight, b1_weight
        else:
            return b0_feat, b0_weight 

    def pers_data_loader(self, use_b1=True):
        '''
        get data_loader for PersNet
        '''
        f_data = self._load_data()
        b0, y = f_data['b0'], f_data['y']
        if use_b1:
            b1 = f_data['b1']
        else:
            b1 = None
        if self.train:
            b0_train, b0_val = self._train_val_split(b0, idx=self.idx)
            y_train, y_val = self._train_val_split(y, idx=self.idx)
            if use_b1:
                b1_train, b1_val = self._train_val_split(b1, idx=self.idx)
            else:
                b1_train, b1_val = None, None
            trainset = self._make_persnet_dataset(y_train, b0_train, b1_train) 
            valset = self._make_persnet_dataset(y_val, b0_val, b1_val) 
            train_loader = data.DataLoader(trainset, **self.loader_params)
            val_loader = data.DataLoader(valset, **self.loader_params)
            return train_loader, val_loader
        else:
            testset = self._make_persnet_dataset(y, b0, b1)
            test_loader = data.DataLoader(testset, **self.loader_params)
            return test_loader

    def data_loader(self, input_dims=None):
        '''
        get a data_loader from table data
        '''
        #b0, b1 = self._load_data()
        f_data = np.load(self.name2load)
        X, y = f_data['X'], f_data['y']        
        if input_dims is not None:
            X = X[:, input_dims]
        if self.train:
            X_train, X_val = self._train_val_split(X, self.idx)
            y_train, y_val = self._train_val_split(y, self.idx)    
            X_train, X_val, y_train, y_val = map(np.array, (X_train, X_val, y_train, y_val))
            trainset, valset = TableDataset(X_train, y_train), TableDataset(X_val, y_val)           
            train_loader = data.DataLoader(trainset, **self.loader_params)
            val_loader = data.DataLoader(valset, **self.loader_params)
            return train_loader, val_loader
        else:
            testset = TableDataset(X, y)
            test_loader = data.DataLoader(testset, **self.loader_params)
            return test_loader                    
