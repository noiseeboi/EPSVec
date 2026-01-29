import numpy as np
from typing import Dict
from dataset_loader import datasets_to_functions, classes_to_labels
from my_utils import load_json
from methods.PromptGenerator import PromptGenerator

def truncate_shots(shots:list, word_truncation:int):
    assert len(shots) == 1, "Currently only supports single shot truncation"
    
    
    filtered_shots = []
    for shot in shots:
        words = shot.split()
        if len(words) > word_truncation:
            print(f"Truncating shot from {len(words)} to {word_truncation} words.")
            truncated_shot = ' '.join(words[:word_truncation])
            filtered_shots.append(truncated_shot + ' ...')
        else:
            filtered_shots.append(shot)
    return filtered_shots

def truncate_to_sentence_tokens(text: str, max_tokens: int, tokenizer) -> str:

    enc = tokenizer(
        text,
        add_special_tokens=False, 
        return_offsets_mapping=True
    )

    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    if len(input_ids) <= max_tokens:
        return text

    last_token_idx = max_tokens - 1
    char_cut = offsets[last_token_idx][1]
    snippet = text[:char_cut]
    last_end = max(snippet.rfind('.'), snippet.rfind('!'), snippet.rfind('?'))
    
    if last_end != -1:
        return snippet[: last_end + 1].rstrip()

    return snippet.rstrip()

def load_binary_pairs(dataset_name:str, per_cluster_count:int, prompt_style:str,
                     neg_data_model_short_name_with_temperature:str, neg_data_source:str, neg_data_seed:int, 
                     neg_data_count:int, n_fixed_shots:int, n_data_shots:int, seed:int, truncation:int, tokenizer:any,
                     neg_data_drop_threshold:int, neg_data_fixed_shots_epsilon:float, neg_data_fixed_shots_delta:float, 
                     clustering_eps:float, clustering_delta:float, n_clusters:int):
    
    assert n_data_shots > 0, "n_data_shots must be greater than 0"
    main_dataset = {}
    
    random_clustering = clustering_eps == 0.0
    assert n_clusters > 1 or random_clustering, "n_clusters must be greater than 1 for clustered data"
    
    if random_clustering:
        md = datasets_to_functions[dataset_name](size = per_cluster_count * n_data_shots * n_clusters, seed = seed, keys=['train'])['train']
        for cls in md.keys():
            main_dataset[cls] = {}
            assert len(md[cls]) == per_cluster_count * n_data_shots * n_clusters, 'Not enough data in random clustering'
            for c in range(n_clusters):
                start_idx = c * per_cluster_count * n_data_shots
                end_idx = (c + 1) * per_cluster_count * n_data_shots
                main_dataset[cls][c] = md[cls][start_idx:end_idx]
    else:
        md = datasets_to_functions[f'c{dataset_name}'](size = per_cluster_count * n_data_shots, seed = seed, k=n_clusters, eps=clustering_eps, delta=clustering_delta, keys=['train'])['train']
        for cls in md.keys():
            main_dataset[cls] = md[cls]
            for c in range(n_clusters):
                assert len(main_dataset[cls][c]) == per_cluster_count * n_data_shots, 'Not enough data in clustered dataset'
    
    
    
    assert neg_data_count >= per_cluster_count * n_data_shots, "Not enough negative data available"
    
    nd = load_json(f'results/{neg_data_model_short_name_with_temperature}/Seed_{neg_data_seed}/{dataset_name}/{prompt_style}/count_{neg_data_count}_qt_{neg_data_drop_threshold}/{neg_data_source}_{n_fixed_shots}_{0}/eps_{neg_data_fixed_shots_epsilon}_delta_{neg_data_fixed_shots_delta}/raw_results.json')
    class_labels_to_ids = classes_to_labels[dataset_name]
    class_labels = list(class_labels_to_ids.keys())
    neg_dataset = {}
    for class_label in class_labels:
        neg_dataset[class_label] = [z['text'] for z in nd if z['class'] == class_label]

    pg:Dict[str, PromptGenerator] = {}
    for class_label in class_labels:
        pg[class_label] = PromptGenerator(dataset_name = dataset_name, class_label=class_label, style=prompt_style, n_shots=n_data_shots, n_fixed_shots=n_fixed_shots, fixed_shots_epsilon=neg_data_fixed_shots_epsilon, fixed_shots_delta=neg_data_fixed_shots_delta)

    data = {'pos': {}, 'neg': {}}

    identifier_tag_start = '<IDTAG>'
    identifier_tag_end = '</IDTAG>'

    for class_label in classes_to_labels[dataset_name]:
        data['pos'][f'{class_label}'] = {}
        for cluster_id in range(n_clusters):

            data['pos'][f'{class_label}'][cluster_id] = []

            aligned_dataset = main_dataset[classes_to_labels[dataset_name][class_label]][cluster_id]

            system_prompt = pg[class_label].get_system_prompt(1)[0]
            assistant_part = pg[class_label].get_assistant_prompt(1)[0]
            
            for i in range(per_cluster_count):
                                
                shots_aligned = aligned_dataset[i * n_data_shots:(i + 1) * n_data_shots]
                
                # shots_aligned = truncate_shots(shots_aligned, word_truncation - len(system_prompt.split()) - len(assistant_part.split()) - 20)
                shots_aligned = [truncate_to_sentence_tokens(shot, truncation, tokenizer) for shot in shots_aligned]
                
                shots_aligned[0] = identifier_tag_start + shots_aligned[0]
                shots_aligned[-1] = shots_aligned[-1] + identifier_tag_end
                
                p = pg[class_label].get_user_prompt(bs = 1, few_shot_samples=shots_aligned)[0]
                
                public_data = p.split(identifier_tag_start)[0].rstrip()
                private_data = ' ' + p.split(identifier_tag_start)[1].split(identifier_tag_end)[0]
                p = p.replace(identifier_tag_start, '').replace(identifier_tag_end, '')
                
                
                this_data = {'system': system_prompt,
                                    'user': p,
                                    'public': public_data,
                                    'private': private_data,
                                    'assistant': assistant_part}

                data['pos'][f'{class_label}'][cluster_id].append(this_data)

        data['neg'][f'{class_label}'] = []
        misaligned_dataset = neg_dataset[class_label]

        
        for i in range(per_cluster_count):
            
            shots_misaligned = misaligned_dataset[i * n_data_shots:(i + 1) * n_data_shots]
            
            shots_misaligned = [truncate_to_sentence_tokens(shot, truncation, tokenizer) for shot in shots_misaligned]

            shots_misaligned[0] = identifier_tag_start + shots_misaligned[0]
            shots_misaligned[-1] = shots_misaligned[-1] + identifier_tag_end

            p = pg[class_label].get_user_prompt(bs = 1, few_shot_samples=shots_misaligned)[0]

            public_data = p.split(identifier_tag_start)[0].rstrip()
            private_data = ' ' + p.split(identifier_tag_start)[1].split(identifier_tag_end)[0]
            p = p.replace(identifier_tag_start, '').replace(identifier_tag_end, '')
            
            this_data = {'system': system_prompt,
                         'user': p,
                         'public': public_data,
                         'private': private_data,
                         'assistant': assistant_part}

            data['neg'][f'{class_label}'].append(this_data)
            
    return data, class_labels_to_ids

if __name__ == "__main__":
    load_binary_pairs('yelp', per_cluster_count=10, n_clusters=5, random_clustering=False, n_fixed_shots=1, n_data_shots=1)
    