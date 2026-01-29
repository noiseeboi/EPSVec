import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from my_utils import save_json, load_json
import os
from functools import partial
from typing import Dict, Mapping

def recursively_subsample(data, size, random_rng, history=''):
    new_dict = {}
    for key in data:
        if isinstance(data[key], list):
            available = len(data[key])
            
            target = size
            if size > available:
                raise ValueError(f"Warning: Requested subsample size {size} is larger than available data size {available} at {history}.{key}.")
                target = available
            
            indices = random_rng.choice(available, target, replace=False).tolist()
            new_dict[key] = [data[key][i] for i in indices]
            
        elif isinstance(data[key], dict):
            new_dict[key] = recursively_subsample(data[key], size, random_rng, history + f'.{key}')
    return new_dict

def subsample(data, size = 0, seed = 0):
    if size <= 0:
        return data
    random_rng = np.random.default_rng(seed)
    data = recursively_subsample(data, size, random_rng, history='')
    return data


def _allocate_proportional_counts(size_map: Mapping[int, int], target_total: int) -> Dict[int, int]:
    """Allocate a global sample budget across buckets without exceeding availability."""
    allocations = {key: 0 for key in size_map}
    total_available = sum(size_map.values())
    if target_total <= 0 or total_available == 0:
        return allocations
    if target_total >= total_available:
        # Need to use all the data
        return {key: size_map[key] for key in size_map}

    # Allocate proportionally.
    raw_allocations = {key: (size_map[key] / total_available) * target_total for key in size_map}
    for key in raw_allocations:
        allocations[key] = min(size_map[key], int(raw_allocations[key]))

    remainder = target_total - sum(allocations.values())
    if remainder <= 0:
        # If we finish allocating, just return. 
        return allocations

    # Knapsack.
    fractional_order = sorted(
        raw_allocations.keys(),
        key=lambda key: raw_allocations[key] - allocations[key],
        reverse=True,
    )
    for key in fractional_order:
        if remainder <= 0:
            break
        if allocations[key] >= size_map[key]:
            continue
        allocations[key] += 1
        remainder -= 1

    if remainder > 0:
        for key in fractional_order:
            if remainder <= 0:
                break
            if allocations[key] >= size_map[key]:
                continue
            take = min(size_map[key] - allocations[key], remainder)
            allocations[key] += take
            remainder -= take

    return allocations


def _allocate_batchwise_counts(size_map: Mapping[int, int], target_total: int, batch_size: int) -> Dict[int, int]:
    """Allocate a sample budget across buckets in batch_size-sized chunks without exceeding availability."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    allocations = {key: 0 for key in size_map}
    available_batches = {key: size_map[key] // batch_size for key in size_map}
    total_available_batches = sum(available_batches.values())
    if target_total <= 0 or total_available_batches == 0:
        return allocations

    target_batches = target_total // batch_size if target_total > 0 else total_available_batches
    batch_allocations = _allocate_proportional_counts(available_batches, target_batches)
    return {key: batch_allocations[key] * batch_size for key in batch_allocations}


def _batchwise_sample_clustered_train(
    train_data: Dict[int, Dict[int, list]], total_size: int, seed: int, batch_size: int
) -> Dict[int, list]:
    """
    Sample per-class clustered data such that each cluster contributes whole batches (batch_size items).
    """
    rng = np.random.default_rng(seed)
    sampled_flat = {}
    for cls_key, clusters in train_data.items():
        cluster_sizes = {cluster_id: len(samples) for cluster_id, samples in clusters.items()}
        allocations = _allocate_batchwise_counts(cluster_sizes, total_size, batch_size)

        flat_list = []
        for cluster_id in sorted(clusters.keys()):
            quota = allocations.get(cluster_id, 0)
            samples = clusters[cluster_id]
            if quota <= 0:
                continue
            if quota >= len(samples):
                chosen = list(samples)
            else:
                indices = rng.choice(len(samples), quota, replace=False).tolist()
                chosen = [samples[i] for i in indices]
            flat_list.extend(chosen)
        sampled_flat[cls_key] = flat_list
    return sampled_flat

# for split in data.keys():
#         for label in data[split]:
#             indices = random_rng.choice(len(data[split][label]), size, replace=False)
#             data[split][label] = [data[split][label][i] for i in indices]

def load_IMDB(size = 0, seed = 0, keys = ['train', 'test']):
    dataset = load_dataset("imdb")
    classes = set(dataset["train"]["label"])

    data = {}
    for split in keys:        
        data[split] = {label: [] for label in classes}
        for example in dataset[split]:
            data[split][example["label"]].append(example['text'])

    data = subsample(data, size, seed)

    return data

def load_AGNews(size = 0, seed = 0, keys = ['train', 'test']):
    dataset = load_dataset("ag_news")
    classes = set(dataset["train"]["label"])

    data = {}
    for split in keys:
        data[split] = {label: [] for label in classes}
        for example in dataset[split]:
            data[split][example["label"]].append(example['text'])

    data = subsample(data, size, seed)
    return data

def load_YELP(size = 0, seed = 0, keys = ['train', 'test']):
    dataset = load_dataset("yelp_polarity")
    classes = set(dataset["train"]["label"])

    data = {}
    for split in keys:
        data[split] = {label: [] for label in classes}
        for example in dataset[split]:
            data[split][example["label"]].append(example['text'])

    data = subsample(data, size, seed)
    return data

def load_biorxiv(size: int = 0, seed: int = 0, keys = ['train', 'test']):
    raw = load_json("my_datasets/biorxiv_filtered_2025-11-29.json")
    all_classes = sorted({row["category"] for rows in raw.values() for row in rows})
    filtered_classes = classes_to_labels['biorxiv']
    # assert that all filtered classes are in all_classes
    assert all(label in all_classes for label in filtered_classes.keys())
    
    # merge test and validation splits into one test split
    if 'validation' in raw:
        raw['test'].extend(raw['validation'])
        del raw['validation']
    
    raw = {split: raw[split] for split in keys}
    data = {split: {label: [] for label in filtered_classes.values()} for split in raw.keys()}
    for split, rows in raw.items():
        for row in rows:
            if row["category"] in filtered_classes:
                data[split][filtered_classes[row["category"]]].append(row["abstract"])

    return subsample(data, size, seed)

def load_openreview(size: int = 0, seed: int = 0, keys = ['train', 'test']):
    raw = load_json("my_datasets/iclr23_reviews.json")
    classes = classes_to_labels['openreview']
    
    raw = {split: raw[split] for split in keys}
    data = {split: {label: [] for label in classes.values()} for split in raw.keys()}
    for split, rows in raw.items():
        for row in rows:
            if row["label"] in classes:
                data[split][classes[row["label"]]].append(row["text"])

    return subsample(data, size, seed)

def load_NYTimes(size = 0, seed = 0):
    assert False, 'There is no test split, fix later'
    dataset = load_dataset("dstefa/New_York_Times_Topics")
    classes = set(dataset["train"]["topic_id"])

    data = {'train': {}, 'test': {}}
    for split in ['train', 'test']:
        data[split] = {label: [] for label in classes}
        for example in dataset[split]:
            data[split][example["topic_id"]].append(example['text'])

    data = subsample(data, size, seed)
    return data


def load_DBPedia(size = 0, seed = 0, keys = ['corpus']):
    assert keys == ['corpus'], "DBPedia loader only supports 'corpus' key currently."
    dataset = load_dataset("mteb/dbpedia", 'corpus')
    data = {}
    data['corpus'] = {}
    data['corpus'][0] = [i for i in range(len(dataset['corpus']))]
    data = subsample(data, size, seed)
    data['corpus'][0] = [dataset['corpus'][i]['text'] for i in data['corpus'][0]]
    return data


def load_clustered_dataset(dataset_name, size = 0, seed = 0, k = 5, eps=1.0, delta=1e-6, keys = ['train']):
    jfile = load_json(f'clustered_data/{dataset_name}/eps_{eps}_delta_{delta}/seed_0/k_{k}/clustered_data.json')
    jfile = {int(key): {int(subkey): jfile[key][subkey] for subkey in jfile[key].keys()} for key in jfile.keys()}
    assert keys == ['train'], "Clustered dataset loader only supports 'train' key currently."
    data = {'train': jfile}
    data = subsample(data, size, seed)
    return data


def load_clustered_dataset_total(dataset_name, batch_group_size: int, size = 0, seed = 0, k = 5, eps=1.0, delta=1e-6):
    """
    Load clustered data but allocate a per-class sample budget proportionally across clusters.
    """
    jfile = load_json(f'clustered_data/{dataset_name}/eps_{eps}_delta_{delta}/seed_0/k_{k}/clustered_data.json')
    jfile = {int(key): {int(subkey): jfile[key][subkey] for subkey in jfile[key].keys()} for key in jfile.keys()}

    data = {'train': jfile}
    # data['test'] = datasets_to_functions[dataset_name](0, seed)['test']

    if size > 0:
        data['train'] = _batchwise_sample_clustered_train(jfile, size, seed, batch_group_size)
    else:
        data['train'] = {cls: {cluster: list(samples) for cluster, samples in clusters.items()} for cls, clusters in jfile.items()}
    return data

classes_to_labels = {
    'imdb': {'negative': 0, 'positive': 1},
    'yelp': {'negative': 0, 'positive': 1},
    'agnews': {'world': 0, 'sports': 1, 'sci-tech': 2, 'business': 3},
    'biorxiv': {
        'neuroscience': 0,
        'microbiology': 1,
        'cell biology': 2,
        'bioinformatics': 3,
    },
    'openreview': {'negative': 0, 'positive': 1}
}

datasets_to_functions = {
    'imdb': load_IMDB,
    'agnews': load_AGNews,
    'yelp': load_YELP,
    'biorxiv': load_biorxiv,
    'openreview': load_openreview,
    'dbpedia': load_DBPedia,
    'cimdb': partial(load_clustered_dataset, dataset_name='imdb'),
    'cyelp': partial(load_clustered_dataset, dataset_name='yelp'),
    'cagnews': partial(load_clustered_dataset, dataset_name='agnews'),
    'cbiorxiv': partial(load_clustered_dataset, dataset_name='biorxiv'),
    'copenreview': partial(load_clustered_dataset, dataset_name='openreview'),
    'cimdb_total': partial(load_clustered_dataset_total, dataset_name='imdb'),
    'cyelp_total': partial(load_clustered_dataset_total, dataset_name='yelp'),
    'cagnews_total': partial(load_clustered_dataset_total, dataset_name='agnews'),
    'cbiorxiv_total': partial(load_clustered_dataset_total, dataset_name='biorxiv'),
    'copenreview_total': partial(load_clustered_dataset_total, dataset_name='openreview')
}



if __name__ == "__main__":
    from my_utils import save_json
    for d in ['agnews', 'yelp', 'biorxiv', 'imdb', 'openreview']:
        data = datasets_to_functions[d](keys=['train', 'test'])
        train_sizes = {k: len(v) for k, v in data['train'].items()}
        test_sizes = {k: len(v) for k, v in data['test'].items()}
        
        save_json({'train_sizes': train_sizes, 'test_sizes': test_sizes}, f'methods/hyperparams/{d}_data_sizes.json')
