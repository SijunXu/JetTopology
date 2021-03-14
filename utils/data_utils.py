import os 
import numpy as np
from tqdm import tqdm
import awkward
#import pickle
import pickle5 as pickle

#from JetTopology.utils import get_p4, round_phi, IRC_cut, JetObs
import JetTopology.utils as utils

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RawData:
    '''

    '''
    def __init__(self, 
    fname='pyhtia_qg_antikt_jetparticle_06_v2.npz',
    path2data='/home/sijun/projects/Topology@Collider/jetTopo/data',
    nb_data=None
    ):
        '''
        get q/g data for [100, 350] GeV for 5 bins, for anti-kt Pythia, CA Pythia and anti-kt Herwig data, take the first `nb_data` data
        '''
        super().__init__()
        self.path2data = path2data
        self.fname = fname
        self.nb_data = nb_data

    def get_data(self):
        
        data = np.load(os.path.join(self.path2data, self.fname))    
        
        if self.fname == 'pyhtia_qg_antikt_jetparticle_06_v2.npz':
            jet_type = 0
        elif self.fname == 'pyhtia_qg_CA_jetparticle_06_v2.npz':
            jet_type = 1
        elif self.fname == 'herwig_qg_antikt_jetparticle_06_v2.npz':
            jet_type = 2        
        logger = logging.getLogger(__name__)
        logger.info('processing data file: ' + self.fname)        
        jet_particle = {}
        data_particle = {}
        for key in data:
            data_particle[key] = data[key].astype(np.float32)
            data_particle[key] = utils.round_phi(data_particle[key])
        data_particle_4p = {}    
        for key in data:
            data_particle_4p[key] = utils.get_p4(data_particle[key])       
            
        logger.info('binning q-g data into 100-350 GeV bins...')     

        jet_particle = {}
        keys = ['q', 'g']
        (a, b) = (50, 100)
        jet_particle[str(a)+'_'+str(b)] = {}
        if jet_type == 0:
            fnames = ['Zq_50_100_1', 'Zg_50_100_1']
        elif jet_type == 1:
            fnames = ['Zq_50_100_1', 'Zg_50_100_1']
        elif jet_type == 2:
            fnames = ['Zq_50_150_1', 'Zg_50_150_1']
        for i in range(2):
            jet_4p = data_particle_4p[fnames[i]].sum()
            masked = data_particle[fnames[i]][(jet_4p.pt>=a)*(jet_4p.pt<b)]
            masked[np.isnan(masked)] = 0
            jet_particle[str(a)+'_'+str(b)][keys[i]] = masked
            logger.info(str(a)+'_'+str(b)+' '+keys[i]+' :'+str(len(masked)))

        (a, b) = (100, 150)
        jet_particle[str(a)+'_'+str(b)] = {}
        if jet_type == 0:
            fnames = ['Zq_100_1', 'Zg_100_1']
        elif jet_type == 1:
            fnames = ['Zq_100_1', 'Zg_100_1']
        elif jet_type == 2:
            fnames = ['Zq_50_150_1', 'Zg_50_150_1']    
        for i in range(2):
            jet_4p = data_particle_4p[fnames[i]].sum()
            masked = data_particle[fnames[i]][(jet_4p.pt>=a)*(jet_4p.pt<b)]
            masked[np.isnan(masked)] = 0
            jet_particle[str(a)+'_'+str(b)][keys[i]] = masked
            logger.info(str(a)+'_'+str(b)+' '+keys[i]+' :'+str(len(masked)))

        (a, b) = (150, 200)
        jet_particle[str(a)+'_'+str(b)] = {}    
        if jet_type == 0:
            fnames = ['Zq_100_1', 'Zg_100_1']
        elif jet_type == 1:
            fnames = ['Zq_100_1', 'Zg_100_1']
        elif jet_type == 2:
            fnames = ['Zq_150_250_1', 'Zg_150_250_1']
        for i in range(2):
            jet_4p = data_particle_4p[fnames[i]].sum()
            masked = data_particle[fnames[i]][(jet_4p.pt>=a)*(jet_4p.pt<b)]
            masked[np.isnan(masked)] = 0
            jet_particle[str(a)+'_'+str(b)][keys[i]] = masked
            logger.info(str(a)+'_'+str(b)+' '+keys[i]+' :'+str(len(masked)))

        (a, b) = (200, 250)
        jet_particle[str(a)+'_'+str(b)] = {}    
        if jet_type == 0:
            fnames = ['Zq_200_300_1', 'Zg_200_300_1']
        elif jet_type == 1:
            fnames = ['Zq_200_300_1', 'Zg_200_300_1']
        elif jet_type == 2:
            fnames = ['Zq_150_250_1', 'Zg_150_250_1']    
        for i in range(2):
            jet_4p = data_particle_4p[fnames[i]].sum()
            masked = data_particle[fnames[i]][(jet_4p.pt>=a)*(jet_4p.pt<b)]
            masked[np.isnan(masked)] = 0
            jet_particle[str(a)+'_'+str(b)][keys[i]] = masked
            logger.info(str(a)+'_'+str(b)+' '+keys[i]+' :'+str(len(masked)))

        (a, b) = (250, 300)
        jet_particle[str(a)+'_'+str(b)] = {}
        if jet_type == 0:
            fnames = ['Zq_200_300_1', 'Zg_200_300_1']
        elif jet_type == 1:
            fnames = ['Zq_200_300_1', 'Zg_200_300_1']
        elif jet_type == 2:
            fnames = ['Zq_250_350_1', 'Zg_250_350_1']    
        for i in range(2):
            jet_4p = data_particle_4p[fnames[i]].sum()
            masked = data_particle[fnames[i]][(jet_4p.pt>=a)*(jet_4p.pt<b)]
            masked[np.isnan(masked)] = 0
            jet_particle[str(a)+'_'+str(b)][keys[i]] = masked
            logger.info(str(a)+'_'+str(b)+' '+keys[i]+' :'+str(len(masked)))

        (a, b) = (300, 350)
        jet_particle[str(a)+'_'+str(b)] = {}
        if jet_type == 0:
            fnames = ['Zq_300_400_1', 'Zg_300_400_1']
        elif jet_type == 1:
            fnames = ['Zq_300_400_1', 'Zg_300_400_1']
        elif jet_type == 2:
            fnames = ['Zq_250_350_1', 'Zg_250_350_1'] 
        for i in range(2):
            jet_4p = data_particle_4p[fnames[i]].sum()
            masked = data_particle[fnames[i]][(jet_4p.pt>=a)*(jet_4p.pt<b)]
            masked[np.isnan(masked)] = 0
            jet_particle[str(a)+'_'+str(b)][keys[i]] = masked        
            logger.info(str(a)+'_'+str(b)+' '+keys[i]+' :'+str(len(masked)))  

        if self.nb_data:
            for key in jet_particle:
                for case in jet_particle[key]:
                    jet_particle[key][case] = jet_particle[key][case][:self.nb_data]

        return jet_particle


#from ..topology import ML_JetPersistance
#import ..topology
from JetTopology import topology

class ML_data:    

    def __init__(
        self, 
        dRmin=1e-2, 
        zeta=1e-2, 
        zeta_type='zeta', 
        R=0.6, 
        graph_type='DT', 
        k=2, 
        p=0,
        path2data='/home/sijun/projects/Topology@Collider/jetTopo/data'
        ):
        super().__init__()
        self.path2data = path2data

        self.dRmin = dRmin
        self.zeta = zeta
        self.zeta_type = zeta_type
        self.R = R
        self.graph_type = graph_type
        ## for kNN based graph
        self.k = k
        self.p = p

    def prepare_ml_data(self):

        ## load raw data
        path2data = self.path2data
        data1 = np.load(os.path.join(path2data, 'pyhtia_qg_antikt_jetparticle_06_run11.npz'))
        data2 = np.load(os.path.join(path2data, 'pyhtia_qg_antikt_jetparticle_06_run12.npz'))
        names1 = list(data1.keys())
        names2 = list(data2.keys())

        raw_data = {}
        for i, key in enumerate(names1):
            raw_data[key] = np.concatenate((data1[key], data2[names2[i]]), axis=0)      
        raw_data_4p = {}
        for key in data1:
            raw_data[key] = utils.round_phi(raw_data[key])
        for key in data1:
            raw_data_4p[key] = utils.get_p4(raw_data[key])

        ## select w.r.t jet pT
        jet_particle = {}
        keys = ['q', 'g']
        for ii in range(5):
            (a, b) = (100 + 50 * ii, 150 + 50 * ii)
            jet_particle[str(a)+'_'+str(b)] = {}
            fnames = [
                'Zq_' + str(a) + '_11', 
                'Zg_' + str(a) + '_11']            
            for i in range(2):
                print('selecting {0:s} data with {1:d} jets...'
                    .format(str(fnames[i]), len(raw_data[fnames[i]])))
                jet_4p = raw_data_4p[fnames[i]].sum()
                masked = raw_data[fnames[i]][ (jet_4p.pt>=a) * (jet_4p.pt<b) ]
                masked[np.isnan(masked)] = 0
                jet_particle[str(a)+'_'+str(b)][keys[i]] = masked                
                logging.info(str(a)+'_'+str(b)+' '+keys[i]+' :'+str(len(masked)))

        jet_particle_4p = {}
        for key in jet_particle:
            jet_particle_4p[key] = {}
            for tt in ['q', 'g']:
                jet_particle_4p[key][tt] = utils.get_p4( jet_particle[key][tt] )         
                logging.info(key + ' ' + tt + ' ' + str(len(jet_particle[key][tt])))        

        max_evt = 22000         
        for key in jet_particle:
            for case in jet_particle[key]:
                pp = jet_particle_4p[key][case]
                jet_particle[key][case] = utils.IRC_cut().process(pp[:max_evt], dRmin=self.dRmin, zeta=self.zeta)
                np.nan_to_num(jet_particle[key][case], copy=False, nan=0.0)
                logging.info(key + ' ' + case + ' ' + str(len(jet_particle[key][case])))
                
        ## split train-test set
        train_jet_particle = {}
        test_jet_particle = {}
        for kk in jet_particle:
            train_jet_particle[kk] = {}
            test_jet_particle[kk] = {}
            for case in jet_particle[kk]:                
                train_jet_particle[kk][case] = jet_particle[kk][case][:20000]
                test_jet_particle[kk][case] = jet_particle[kk][case][20000:]    

        train_jet_particle['all'] = {}        
        for key in ['q', 'g']:
            train_jet_particle['all'][key] = np.concatenate((
                train_jet_particle['100_150'][key],
                train_jet_particle['150_200'][key], train_jet_particle['200_250'][key],
                train_jet_particle['250_300'][key], train_jet_particle['300_350'][key]
            ), axis=0) 

        test_jet_particle['all'] = {}        
        for key in ['q', 'g']:
            test_jet_particle['all'][key] = np.concatenate((
                test_jet_particle['100_150'][key],
                test_jet_particle['150_200'][key], test_jet_particle['200_250'][key],
                test_jet_particle['250_300'][key], test_jet_particle['300_350'][key]
            ), axis=0)   
        return train_jet_particle['all'], test_jet_particle['all']

    def get_DT_ml_data(self, name2save=None):
        train_jet_particle, test_jet_particle = self.prepare_ml_data()
        ## compute persistence information
        train_b0_pair = {}
        train_b1_pair = {}
        for key in train_jet_particle:            
            pers_pairs = topology.ML_JetPersistance().get_ml_inputs(
                utils.get_p4(train_jet_particle[key]), 
                zeta_type=self.zeta_type, 
                R=self.R
                )
            train_b0_pair[key], train_b1_pair[key] = pers_pairs['b0'], pers_pairs['b1']

        test_b0_pair = {}
        test_b1_pair = {}
        for key in test_jet_particle:
            pers_pairs = topology.ML_JetPersistance().get_ml_inputs(
                utils.get_p4(test_jet_particle[key]), 
                zeta_type=self.zeta_type, 
                R=self.R
                )
            test_b0_pair[key], test_b1_pair[key] = pers_pairs['b0'], pers_pairs['b1']

        ml_data = {
            'train_b0': train_b0_pair,
            'train_b1': train_b1_pair,            
            'test_b0': test_b0_pair,
            'test_b1': test_b1_pair
        }
        
        if not name2save:
            return ml_data
        else:
            with open(name2save, 'wb') as handle:
                pickle.dump(ml_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def get_jet_obs(self, name2save=None):
        '''
        compute N-Subjettiness for n=1-6
        '''
        train_jet_particle, test_jet_particle = self.prepare_ml_data()
        train_obs, test_obs = {}, {}
        Ns = [1, 2, 3, 4, 5, 6]
        for key in train_jet_particle:
            taus = []
            for N in Ns:
                taus.append(utils.JetObs().Njettiness(utils.get_p4(train_jet_particle[key]), N=N, beta=0.2))
            train_obs[key] = np.vstack(taus).T
        for key in test_jet_particle:
            taus = []
            for N in Ns:
                taus.append(utils.JetObs().Njettiness(utils.get_p4(test_jet_particle[key]), N=N, beta=0.2))
            test_obs[key] = np.vstack(taus).T
        obs = {
            'train': train_obs,
            'test': test_obs
        }        
        if not name2save:
            return obs
        else:
            with open(name2save, 'wb') as handle:
                pickle.dump(obs, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def get_kNN_ml_data(self, name2save=None):
        train_jet_particle, test_jet_particle = self.prepare_ml_data()
        ## compute persistence information for b0 features based on kNN graphs
        train_b0_pair = {}
        for key in train_jet_particle:            
            train_b0_pair[key] = topology.ML_JetPersistance().get_kNN_ml_inputs(
                utils.get_p4(train_jet_particle[key]), 
                k=self.k, 
                p=self.p)

        test_b0_pair = {}        
        for key in test_jet_particle:
            test_b0_pair[key] = topology.ML_JetPersistance().get_kNN_ml_inputs(
                utils.get_p4(test_jet_particle[key]), 
                k=self.k,
                p=self.p)            

        ml_data = {
            'train_b0': train_b0_pair,            
            'test_b0': test_b0_pair,            
        }
        
        if not name2save:
            return ml_data
        else:
            with open(name2save, 'wb') as handle:
                pickle.dump(ml_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_ml_data(self, name2load=None):
        if name2load:
            with open(name2load, 'rb') as handle:
                ml_data = pickle.load(handle)
            return ml_data    

    