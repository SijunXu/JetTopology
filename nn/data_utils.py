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


class JetData:    
    r'''
    convert `train_b0`, `train_b1`, `test_b0`, `test_b1` to `DataLoader` object
    '''
    def __init__(self, train=True, idx=None, name2load=None, loader_params=None):
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

    def _load_data(self):      
        '''
        load pickle data into `b0` and `b1`
        '''  
        with open(self.name2load, 'rb') as handle:
            pers = pickle.load(handle)
        if self.train:
            return pers['train_b0'], pers['train_b1']
        else:
            return pers['test_b0'], pers['test_b1']
    
    def _load_kNN_data(self):
        with open(self.name2load, 'rb') as handle:
            pers = pickle.load(handle)
        if self.train:
            return pers['train_b0']
        else:
            return pers['test_b0']
        

    def _cat2table(self, b0, b1=None):
        r'''
        input a list of b0 persistence pairs and b1 pairs, concatenate them into arrays         
        for example case of [100, 350] GeV data, we choose first 6 b0 features and first 5 b0 features
        in total 6 * 5 + 5 * 4 = 50 dim, zero padded
        '''
        n_jets = len(b0)
        max_b0, max_b1 = 6, 5
        if b1 is None:
            max_b1 = 0
            b1 = [[] for _ in range(n_jets)]        
        table = np.zeros((n_jets, int(max_b0 * 5 + max_b1 * 4)))

        for i, (p0, p1) in enumerate( zip(b0, b1) ):
            n_b0, n_b1 = len(p0), len(p1)
            b0_info = []        
            for p in p0[:max_b0]:
                b0_info += list(p)
            table[i][:(min(max_b0, n_b0) * 5)] = b0_info

            if n_b1 > 0:
                b1_info = []
                for p in p1[:max_b1]:
                    b1_info += list(p)
                table[i][(max_b0 * 5) : (max_b0 * 5 + min(max_b1, n_b1) * 4)] = b1_info
        return table

    def _cat_persnet_feat(self, b0, b1, max_b0=None, max_b1=None):
        n_jets = len(b0)        
        if max_b0 is None:
            max_b0 = 0
            for p0 in b0:
                max_b0 = max(len(p0), max_b0)
        
        if max_b1 is None:
            max_b1 = 0
            for p1 in b1:
                max_b1 = max(len(p1), max_b1)
        
        b0_feat, b1_feat = np.zeros((n_jets, max_b0, 5)), np.zeros((n_jets, max_b1, 4))
        b0_weight, b1_weight = np.zeros((n_jets, max_b0)), np.zeros((n_jets, max_b1))

        for i, p0 in enumerate(b0):
            n_b0 = len(p0)
            if n_b0 == 1:
                b0_feat[i, 0] = list(p0[0])
            elif n_b0 > 1:
                b0_feat[i, :n_b0] = p0
        for i, p1 in enumerate(b1):
            n_b1 = len(p1)
            if n_b1 == 1:
                b1_feat[i, 0] = list(p1[1])
            elif n_b1 > 1:
                b1_feat[i, :n_b1] = p1

        ## weight = log(b) - log(d)
        b0_weight = b0_feat[:, :, 0] - b0_feat[:, :, 1]
        b1_weight = b1_feat[:, :, 1] - b1_feat[:, :, 0]
        return b0_feat, b1_feat, b0_weight, b1_weight

    def _cat_kNN_persnet_feat(self, b0,max_b0=None):
        n_jets = len(b0)        
        if max_b0 is None:
            max_b0 = 0
            for p0 in b0:
                max_b0 = max(len(p0), max_b0)        
        
        b0_feat = np.zeros((n_jets, max_b0, 5))
        b0_weight = np.zeros((n_jets, max_b0))

        for i, p0 in enumerate(b0):
            n_b0 = len(p0)
            if n_b0 == 1:
                b0_feat[i, 0] = list(p0[0])
            elif n_b0 > 1:
                b0_feat[i, :n_b0] = p0 
        ## weight = log(b) - log(d)
        b0_weight = b0_feat[:, :, 0] - b0_feat[:, :, 1]        
        return b0_feat, b0_weight

    
    def _train_val_split(self, train_data, kfold=5, idx=0):
        r'''
        input `train_data` as a dict with keys as 'q', 'g'
        returns `train_data_tra` and `train_data_val` for 5 folds
        '''
        nb_val = int( 20000/kfold )        
        train_data_tra = {}
        train_data_val = {}
        for key in train_data:
            train_data_tra[key] = []
            train_data_val[key] = []
            for i in range(5):
                '''
                loop over 5 pt bins 
                '''
                full_idx = list(range(i*20000, (i+1)*20000))            
                val_idx = list(range(i*20000+idx*nb_val, i*20000+(idx+1)*nb_val))
                train_idx = list(set(full_idx)-set(val_idx))

                train_data_tra[key] += [ train_data[key][idx] for idx in train_idx ]
                train_data_val[key] += [ train_data[key][idx] for idx in val_idx ]                    

        return train_data_tra, train_data_val

    def _make_dataset(self, b0, b1):
        
        def pad_cat(a, b):
            shape_a, shape_b = a.shape, b.shape

            if len(shape_a) == 3:
                dim_cat = max(shape_a[1], shape_b[1])
                X = np.zeros((shape_a[0] + shape_b[0], dim_cat, shape_a[-1]))
                X[:shape_a[0], :shape_a[1]] = a
                X[shape_a[0]:, :shape_b[1]] = b
                return X

            elif len(shape_a) == 2:
                dim_cat = max(shape_a[1], shape_b[1])
                X = np.zeros((shape_a[0] + shape_b[0], dim_cat))
                X[:shape_a[0], :shape_a[1]] = a
                X[shape_a[0]:, :shape_b[1]] = b
                return X

        feats = []
        for key in ['q', 'g']:
            #b0_feat, b1_feat, b0_weight, b1_weight = self._cat_persnet_feat(b0[key], b1[key])            
            feats.append(self._cat_persnet_feat(b0[key], b1[key]))            
        b0_feat = pad_cat(feats[0][0], feats[1][0])
        b1_feat = pad_cat(feats[0][1], feats[1][1])
        b0_weight = pad_cat(feats[0][2], feats[1][2])
        b1_weight = pad_cat(feats[0][3], feats[1][3])
        
        y = [1 for _ in range(len(feats[0][0]))] + [0 for _ in range(len(feats[1][0]))]
        return PersDataset(b0_feat, b0_weight, b1_feat, b1_weight, y)
    
    def _make_kNN_dataset(self, b0):
        def pad_cat(a, b):
            shape_a, shape_b = a.shape, b.shape

            if len(shape_a) == 3:
                dim_cat = max(shape_a[1], shape_b[1])
                X = np.zeros((shape_a[0] + shape_b[0], dim_cat, shape_a[-1]))
                X[:shape_a[0], :shape_a[1]] = a
                X[shape_a[0]:, :shape_b[1]] = b
                return X

            elif len(shape_a) == 2:
                dim_cat = max(shape_a[1], shape_b[1])
                X = np.zeros((shape_a[0] + shape_b[0], dim_cat))
                X[:shape_a[0], :shape_a[1]] = a
                X[shape_a[0]:, :shape_b[1]] = b
                return X 

        feats = []
        for key in ['q', 'g']:            
            feats.append( self._cat_kNN_persnet_feat(b0[key]) )
        b0_feat = pad_cat(feats[0][0], feats[1][0])        
        b0_weight = pad_cat(feats[0][1], feats[1][1])
        
        y = [1 for _ in range(len(feats[0][0]))] + [0 for _ in range(len(feats[1][0]))]
        return PerskNNDataset(b0_feat, b0_weight, y)


    def pers_data_loader(self):
        '''
        get data_loader for PersNet
        '''

        b0, b1 = self._load_data()
        if self.train:
            b0_train, b0_val = self._train_val_split(b0, idx=self.idx)
            b1_train, b1_val = self._train_val_split(b1, idx=self.idx)

            trainset = self._make_dataset(b0_train, b1_train) 
            valset = self._make_dataset(b0_val, b1_val)

            train_loader = data.DataLoader(trainset, **self.loader_params)
            val_loader = data.DataLoader(valset, **self.loader_params)
            return train_loader, val_loader

        else:
            testset = self._make_dataset(b0, b1)
            test_loader = data.DataLoader(testset, **self.loader_params)
            return test_loader
    
    def kNN_pers_data_loader(self):
        '''
        get data_loader for PersNet for kNN based graph
        '''
        b0 = self._load_kNN_data()
        if self.train:
            b0_train, b0_val = self._train_val_split(b0, idx=self.idx)
            trainset = self._make_kNN_dataset(b0_train) 
            valset = self._make_kNN_dataset(b0_val)

            train_loader = data.DataLoader(trainset, **self.loader_params)
            val_loader = data.DataLoader(valset, **self.loader_params)
            return train_loader, val_loader

        else:
            testset = self._make_kNN_dataset(b0)
            test_loader = data.DataLoader(testset, **self.loader_params)
            return test_loader
    
    def kNN_data_loader(self):
        b0 = self._load_kNN_data()
        if self.train:
            b0_train, b0_val = self._train_val_split(b0, idx=self.idx)
            sig_train = self._cat2table(b0_train['q'])
            bg_train = self._cat2table(b0_train['g'])
            X_train = np.concatenate((sig_train, bg_train), axis=0)
            y_train = [1 for _ in range(len(sig_train))] + [0 for _ in range(len(bg_train))]

            sig_val = self._cat2table(b0_val['q'])
            bg_val = self._cat2table(b0_val['g'])
            X_val = np.concatenate((sig_val, bg_val), axis=0)
            y_val = [1 for _ in range(len(sig_val))] + [0 for _ in range(len(bg_val))]

            trainset, valset = TableDataset(X_train, y_train), TableDataset(X_val, y_val)      

            train_loader = data.DataLoader(trainset, **self.loader_params)
            val_loader = data.DataLoader(valset, **self.loader_params)
            return train_loader, val_loader
        else:
            sig = self._cat2table(b0['q'])
            bg = self._cat2table(b0['g'])
            X = np.concatenate((sig, bg), axis=0)
            y = [1 for _ in range(len(sig))] + [0 for _ in range(len(bg))]
            testset = TableDataset(X, y)
            test_loader = data.DataLoader(testset, **self.loader_params)
            return test_loader     



    def data_loader(self):
        b0, b1 = self._load_data()
        if self.train:
            b0_train, b0_val = self._train_val_split(b0, idx=self.idx)
            b1_train, b1_val = self._train_val_split(b1, idx=self.idx)

            sig_train = self._cat2table(b0_train['q'], b1_train['q'])
            bg_train = self._cat2table(b0_train['g'], b1_train['g'])
            X_train = np.concatenate((sig_train, bg_train), axis=0)

            sig_val = self._cat2table(b0_val['q'], b1_val['q'])
            bg_val = self._cat2table(b0_val['g'], b1_val['g'])
            X_val = np.concatenate((sig_val, bg_val), axis=0)

            y_train = [1 for _ in range(len(sig_train))] + [0 for _ in range(len(bg_train))]
            y_val = [1 for _ in range(len(sig_val))] + [0 for _ in range(len(bg_val))]
            
            trainset, valset = TableDataset(X_train, y_train), TableDataset(X_val, y_val)           

            train_loader = data.DataLoader(trainset, **self.loader_params)
            val_loader = data.DataLoader(valset, **self.loader_params)
            return train_loader, val_loader

        else:
            sig = self._cat2table(b0['q'], b1['q'])
            bg = self._cat2table(b0['g'], b1['g'])
            X = np.concatenate((sig, bg), axis=0)
            y = [1 for _ in range(len(sig))] + [0 for _ in range(len(bg))]
            testset = TableDataset(X, y)
            test_loader = data.DataLoader(testset, **self.loader_params)
            return test_loader            