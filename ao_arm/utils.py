import torch
import numpy as np
import selfies as sf

def image_int_to_float(x, binary=False):
    if binary:
        return x.float() * 2 - 1.0 # from {0, 1} to {-1.0, 1.0}
    else:
        return x / 127.5 - 1.

def image_float_to_int(x, binary=False):
    if binary:
        return torch.round( (x+1.0)/2.0 ).long() # from {-1.0, 1.0} to {0, 1}
    else:
        return torch.round( (x+1) * 127.5 ).long()

def text_int_to_str(x, with_mask=False):
    mp = [chr(ord('a') + i) for i in range(26)]
    if with_mask:
        mp = ['_', ' '] + mp
    else:
        mp = [' '] + mp

    return [''.join([mp[t] for t in row]) for row in x]

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

def multiple_selfies_to_int(selfies_list, largest_molecule_len, alphabet):
    """Convert a list of selfies strings to integer encodings
    """
    int_list = []
    for s in selfies_list:
        integer_encoded = selfies_to_int(s, largest_molecule_len, alphabet)
        int_list.append(integer_encoded)
    return np.array(int_list)

def indices_to_selfies(indices, alphabet):
    """Convert list of indices to selfies string
    """
    int_to_symbol = dict((i, c) for i, c in enumerate(alphabet))
    selfie = ''
    for num in indices:
        selfie += int_to_symbol[num]
    return selfie

def multiple_indices_to_selfies(indices_ls, alphabet):
    """Convert multiple lists of indices to selfies string
    """
    selfies_list = []
    for indices in indices_ls:
        selfies_list.append(indices_to_selfies(indices, alphabet))
    return np.array(selfies_list)

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