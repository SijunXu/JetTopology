import os 
import numpy as np
from tqdm import tqdm
import awkward
import pickle

from ..topology import ML_JetPersistance, JetPersistance
from . import get_p4, round_phi, IRC_cut

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ML_data:

    def __init__(self, dRmin=1e-2, zeta=1e-2, zeta_type='zeta', R=0.6):
        super().__init__()
        self.dRmin = dRmin
        self.zeta = zeta
        self.zeta_type = zeta_type
        self.R = R

    def prepare_ml_data(self, name2save=None):

        ## load raw data
        path2data = '/home/sijun/projects/Topology@Collider/jetTopo/data'
        data1 = np.load(os.path.join(path2data, 'pyhtia_qg_antikt_jetparticle_06_run11.npz'))
        data2 = np.load(os.path.join(path2data, 'pyhtia_qg_antikt_jetparticle_06_run12.npz'))
        names1 = list(data1.keys())
        names2 = list(data2.keys())

        raw_data = {}
        for i, key in enumerate(names1):
            raw_data[key] = np.concatenate((data1[key], data2[names2[i]]), axis=0)      
        raw_data_4p = {}
        for key in data1:
            raw_data[key] = round_phi(raw_data[key])
        for key in data1:
            raw_data_4p[key] = get_p4(raw_data[key])

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
                jet_particle_4p[key][tt] = get_p4( jet_particle[key][tt] )         
                logging.info(key + ' ' + tt + ' ' + str(len(jet_particle[key][tt])))        

        max_evt = 22000         
        for key in jet_particle:
            for case in jet_particle[key]:
                pp = jet_particle_4p[key][case]
                jet_particle[key][case] = IRC_cut().process(pp[:max_evt], dRmin=self.dRmin, zeta=self.zeta)
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
            
        ## compute persistence information
        train_b0_pair = {}
        train_b1_pair = {}
        for key in train_jet_particle['all']:            
            pers_pairs = ML_JetPersistance().get_ml_inputs(
                get_p4(train_jet_particle['all'][key]), 
                zeta_type=self.zeta_type, 
                R=self.R
                )
            train_b0_pair[key], train_b1_pair[key] = pers_pairs['b0'], pers_pairs['b1']

        test_b0_pair = {}
        test_b1_pair = {}
        for key in test_jet_particle['all']:
            pers_pairs = ML_JetPersistance().get_ml_inputs(
                get_p4(test_jet_particle['all'][key]), 
                zeta_type=self.zeta_type, 
                R=self.R
                )
            test_b0_pair, test_b1_pair = pers_pairs['b0'], pers_pairs['b1']

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
    
    def load_ml_data(self, name2load=None):
        if name2load:
            with open(name2load, 'rb') as handle:
                ml_data = pickle.load(handle)
            return ml_data    

    