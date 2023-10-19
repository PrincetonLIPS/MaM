import seaborn as sns
from matplotlib import pyplot as plt

import torch

from functools import partial
from multiprocessing import Pool

from rdkit.Chem.QED import qed
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MolToSmiles
from rdkit.Chem import Draw
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol as finger
from rdkit.DataStructs import FingerprintSimilarity as finger_sim
from rdkit.Chem.Descriptors import MolLogP as logP
from rdkit import RDLogger
import selfies as sf

from utils.mol_utils import indices_to_string
from utils.sascorer import calculateScore as sas

NUM_CPU_CORES = 8

def my_get_mol(smiles):
    mol = MolFromSmiles(smiles)
    return mol

def get_san_smiles(mol):
    Chem.SanitizeMol(mol)
    return MolToSmiles(mol, isomericSmiles=False)

def remove_stereo(smiles):
    mol = my_get_mol(smiles)
    return get_san_smiles(mol)

def my_qed(m):
    try:
        score = qed(m)
    except:
        print('Score calculation failed!')
        score = None
    return score

def my_sas(m):
    try:
        score = sas(m)
    except:
        print('Score calculation failed!')
        score = None
    return score

def my_logP(m):
    try:
        score = logP(m)
    except:
        print('Score calculation failed!')
        score = None
    return score

def get_score_from_smiles(smiles, metric_f):
    try:
        smiles = remove_stereo(smiles)
        mol = my_get_mol(smiles)
        score = metric_f(mol)
    except Exception as e:
        score = - 1e10
    return score


METRICS = [('QED', my_qed, 4.0), ('SA', my_sas, 0.6), ('logP', my_logP, 0.35)]

METRICS_FUNCTIONS = {
    'QED': my_qed,
    'SA': my_sas,
    'logP': my_logP,
}

class MolEvalModel(object):
    def __init__(self, alphabet, string_type, metric_name, target_value, tau=1.0):
        self.alphabet = alphabet
        self.string_type = string_type
        self.metric_name = metric_name
        self.target_value = target_value
        self.metric_f = METRICS_FUNCTIONS[metric_name]
        self.tau = tau

    def __call__(self, x):
        val_list = x.tolist()
        p = Pool(NUM_CPU_CORES)
        string_list = p.map(partial(indices_to_string, alphabet=self.alphabet), val_list)
        if self.string_type == 'SELFIES':
            string_list = p.map(sf.decoder, string_list)
        RDLogger.DisableLog('rdApp.*') 
        scores = p.map(partial(get_score_from_smiles, metric_f=self.metric_f), string_list)
        scores = torch.Tensor(scores)
        rewards = - (scores - self.target_value)**2 / self.tau
        return rewards.to(x.device)
    
    def get_scores(self, x):
        val_list = x.tolist()
        p = Pool(NUM_CPU_CORES)
        string_list = p.map(partial(indices_to_string, alphabet=self.alphabet), val_list)
        if self.string_type == 'SELFIES':
            string_list = p.map(sf.decoder, string_list)
        RDLogger.DisableLog('rdApp.*') 
        scores = p.map(partial(get_score_from_smiles, metric_f=self.metric_f), string_list)
        scores = torch.Tensor(scores)
        return scores.to(x.device)
    
    def plot_scores(self, properties, save_path):
        sns.displot(properties, kind="kde")
        plt.savefig(save_path)
        plt.close()