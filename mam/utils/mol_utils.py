"""
This file is to encode SMILES and SELFIES into one-hot encodings
"""

import numpy as np
import selfies as sf
import os

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw

def val_to_selfiles(val, alphabet):
    """
    Convert a single value to a selfies string
    """
    selfies = ''
    for i in val:
        selfies += alphabet[i]
    return selfies

def multiple_val_to_selfies(val_np_arr, alphabet):
    """
    Convert a list of values to a list of selfies strings
    """
    selfies_list = []
    val_np_arr = val_np_arr - 1.0
    val_list = val_np_arr.astype(int).tolist()
    for val in val_list:
        selfies_list.append(val_to_selfiles(val, alphabet))
    return selfies_list

def smiles_to_hot(smile, largest_smile_len, alphabet):
    """
    Convert a single smile string to a one-hot encoding.
    """
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # pad with ' '
    smile += ' ' * (largest_smile_len - len(smile))
    # integer encode input smile
    integer_encoded = [char_to_int[char] for char in smile]
    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)


def multiple_smiles_to_hot(smiles_list, largest_molecule_len, alphabet):
    """
    Convert a list of smile strings to a one-hot encoding
    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """
    hot_list = []
    for smile in smiles_list:
        _, onehot_encoded = smiles_to_hot(smile, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)


def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """
    Convert a single selfies string to a one-hot encoding.
    """
    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))
    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]
    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)

def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """Convert a list of selfies strings to a one-hot encoding
    """
    hot_list = []
    for s in selfies_list:
        _, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)

def smiles_to_int(smile, largest_smile_len, alphabet):
    """
    Convert a single smile string to integer encoding
    """
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # pad with ' '
    smile += ' ' * (largest_smile_len - len(smile))
    # integer encode input smile
    integer_encoded = [char_to_int[char] for char in smile]
    return integer_encoded

def multiple_smiles_to_int(smiles_list, largest_molecule_len, alphabet):
    """
    Convert a list of smile strings to integer encoding
    """
    int_list = []
    for smile in smiles_list:
        integer_encoded = smiles_to_int(smile, largest_molecule_len, alphabet)
        int_list.append(integer_encoded)
    return np.array(int_list)

def selfies_to_int(selfie, largest_selfie_len, alphabet):
    """
    Convert a single selfies string to integer encoding
    """
    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))
    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]
    return integer_encoded

def string_to_int(string, string_type, largest_len, alphabet):
    if string_type == 'SMILES':
        return smiles_to_int(string, largest_len, alphabet)
    elif string_type == 'SELFIES':
        return selfies_to_int(string, largest_len, alphabet)
    else:
        raise ValueError('Invalid string type')


def multiple_selfies_to_int(selfies_list, largest_molecule_len, alphabet):
    """Convert a list of selfies strings to integer encodings
    """
    int_list = []
    for s in selfies_list:
        integer_encoded = selfies_to_int(s, largest_molecule_len, alphabet)
        int_list.append(integer_encoded)
    return np.array(int_list)

def indices_to_string(indices, alphabet):
    """Convert list of indices to smiles string
    """
    int_to_symbol = dict((i, c) for i, c in enumerate(alphabet))
    string = ''
    for num in indices:
        if num == -1:
            string += '[?]' # for case when we put a padding of -1 for ? symbol
        elif int_to_symbol[num] == '[nop]':
            string += ''    
        else:
            string += int_to_symbol[num]
    return string

def multiple_indices_to_string(indices_ls, alphabet):
    """Convert multiple lists of indices to selfies string
    """
    string_list = []
    for indices in indices_ls:
        string_list.append(indices_to_string(indices, alphabet))
    return np.array(string_list)

def get_alphabet_from_smiles(smiles_list):
    """Constructs an alphabet from an iterable of SMILES strings.
    The returned alphabet is the set of all symbols that appear in the
    SMILES strings from the input iterable, minus the dot ``.`` symbol.
    :param smiles_list: an list of SMILES strings.
    :return: an alphabet of SMILES symbols, built from the input list.
    """
    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding
    return smiles_alphabet

def draw_mol_to_file(mol_lst, directory):
    """Saves the pictorial representation of each molecule in the list to
    file."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    for smiles in mol_lst:
        mol = MolFromSmiles(smiles)
        Draw.MolToFile(mol,directory+'/'+smiles+'.pdf')