import numpy as np
import torch
from pyhealth.tokenizer import Tokenizer
from tqdm import tqdm

def flatten(nested_list):
    """
    Flatten a nested list of arbitrary depth to a flat list.
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def multihot(label_indices, num_labels):
    """
    Convert a list of label indices to a multi-hot numpy vector.
    """
    vec = np.zeros(num_labels, dtype=int)
    for idx in label_indices:
        vec[idx] = 1
    return vec

def prepare_label(sample_dataset, drugs):
    """
    Prepare multi-hot label for drug recommendation.
    """
    tokenizer = Tokenizer(sample_dataset.get_all_tokens(key='drugs'))
    indices = tokenizer.convert_tokens_to_indices(drugs)
    num_labels = tokenizer.get_vocabulary_size()
    return multihot(indices, num_labels)

def prepare_drug_indices(sample_dataset):
    """
    Annotate each patient with 'drugs_ind' tensor (multi-hot).
    """
    for patient in tqdm(sample_dataset, desc="Preparing drug indices"):
        label = prepare_label(sample_dataset, patient['drugs'])
        patient['drugs_ind'] = torch.tensor(label, dtype=torch.float)
    return sample_dataset

def pad_and_convert(visits, max_visits, max_nodes):
    """
    Given a list of node-code lists (one per visit),
    produce a tensor of shape [max_visits, max_nodes] with multi-hot encoding,
    reversed in time (most recent first) and padded with zeros.
    """
    padded = []
    for codes in reversed(visits):
        vec = torch.zeros(max_nodes, dtype=torch.float)
        for code in codes:
            vec[code] = 1
        padded.append(vec)
    while len(padded) < max_visits:
        padded.append(torch.zeros(max_nodes, dtype=torch.float))
    return torch.stack(padded, dim=0)

def preprocess_samples(dataset):
    """
    Preprocess all samples in the given dataset (iterable) and return a list.
    """
    samples = []
    for record in tqdm(dataset, desc="Preprocessing samples"):
        samples.append(record)
    return samples

