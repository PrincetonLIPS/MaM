import torch
import torch.utils.data as data_utils
import pandas as pd
import numpy as np

import os
import json

import moses
import selfies as sf

from utils.mol_utils import multiple_selfies_to_int, multiple_smiles_to_int, get_alphabet_from_smiles


def load_molecules_from_smiles(smiles_list, alphabet, largest_len, option):
    if alphabet is None or largest_len is None:
        raise ValueError('alphabet and largest selfies length must be provided')
    if option == 'SELFIES':
        print('--> Translating SMILES to SELFIES...')
        selfies_list = list(map(sf.encoder, smiles_list))
        print('--> Translating SELFIES to INT...')
        data = multiple_selfies_to_int(selfies_list, largest_len, alphabet) # --> {0,1,..,K-1}
    elif option == 'SMILES':
        print('--> Translating SMILES to INT...')
        data = multiple_smiles_to_int(smiles_list, largest_len, alphabet)
    else:
        raise ValueError('option must be either SELFIES or SMILES')
    data = torch.tensor(data + 1, dtype=torch.float32) # {0,1,..,K-1} --> {1,2,..,K}
    y = torch.zeros(data.shape[0], 1) # idle y's
    dataset = data_utils.TensorDataset(data, y)
    return dataset

def get_alphabet(smiles_list, alphabet_path, option):
    print('--> Start constructing alphabet...')
    if option == 'SELFIES':
        print('--> Translating SMILES to SELFIES...')
        selfies_list = list(map(sf.encoder, smiles_list))
        print('--> Constructing SELFIES alphabet...')
        all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
        all_selfies_symbols.add('[nop]')
        alphabet = list(all_selfies_symbols)
        largest_len = max(sf.len_selfies(s) for s in selfies_list)
    elif option == 'SMILES':
        print('--> Constructing SMILES alphabet...')
        alphabet = get_alphabet_from_smiles(smiles_list)
        largest_len = len(max(smiles_list, key=len))
    alphabet_size = len(alphabet)
    alphabet = {
        'alphabet': alphabet, 
        'largest_len': largest_len, 
        'alphabet_size': alphabet_size,
    }
    print('--> Alphabet constructed. Saving to {}...'.format(alphabet_path))
    with open(alphabet_path, 'w') as f:
        json.dump(alphabet, f)
    return alphabet

def load_alphabet(smiles_list, alphabet_path, option):
    if os.path.exists(alphabet_path):
        with open(alphabet_path, 'r') as f:
            alphabet_dict = json.load(f)
    else:
        alphabet_dict = get_alphabet(smiles_list, alphabet_path, option=option)
    alphabet = alphabet_dict['alphabet']
    largest_len = alphabet_dict['largest_len']
    alphabet_size = alphabet_dict['alphabet_size']
    return alphabet, largest_len, alphabet_size

def load_smiles_from_file(file_path):
    df = pd.read_csv(file_path, header=None)
    smiles_list = np.asanyarray(df[0])
    return smiles_list

def load_moses(config):
    print('--> Load moses data...')
    moses_alphabet_path = os.path.join(config.data_dir, 'moses', f'moses_alphabet_{config.string_type}.json')
    all_smiles_list = moses.get_dataset('train')
    test_smiles_list = moses.get_dataset('test')
    alphabet, largest_len, alphabet_size = load_alphabet(all_smiles_list, moses_alphabet_path, config.string_type)
    print('--> alphabet size: {}, largest length: {}'.
        format(alphabet_size, largest_len))
    config.K = alphabet_size
    if not isinstance(config.L, (float, int)):
        config.L = largest_len
    else:
        config.L = int(config.L)
    config.alphabet = alphabet
    # do a 90-10 train-val split
    num_train = int(len(all_smiles_list) * 0.9)
    if config.load_full:
        train_smiles_list = all_smiles_list[:num_train]
        val_smiles_list = all_smiles_list[num_train:]
    else:
        train_smiles_list = all_smiles_list[:10000]
        val_smiles_list = all_smiles_list[10000:11000]  
        test_smiles_list = test_smiles_list[:1000]
    dataset_train = load_molecules_from_smiles(train_smiles_list, alphabet, config.L, config.string_type)
    dataset_val = load_molecules_from_smiles(val_smiles_list, alphabet, config.L, config.string_type)
    dataset_test = load_molecules_from_smiles(test_smiles_list, alphabet, config.L, config.string_type)
    return dataset_train, dataset_val, dataset_test




