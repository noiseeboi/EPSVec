# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# %%
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from steer_vec_utils import (TextDataset, probe_classification, extract_hidden_states, probe_regression, seed_everywhere)
import argparse
from concept_datasets import load_binary_pairs
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from copy import deepcopy
from my_utils import load_json, str2bool

# %%
def get_args(run_in_notebook = False):

    parser = argparse.ArgumentParser()

    parser.add_argument("--bs", type=int, default=1, help="Model forward batch size")
    parser.add_argument("--model_name", type=str, help="Model Name", default="meta-llama/Llama-3.1-8B")
    
    parser.add_argument("--use_quantization", action='store_true', help="Use quantization for the model", default=False)
    
    parser.add_argument("--dataset", type=str, default='yelp', help="Dataset name for binary pairs generation")
    parser.add_argument("--target_tokens", type=str, default='all', choices=['last', 'all', 'assistant'], help="Type of vector to use for the concept; meandiff only applies to last token, meandiffall applies to all tokens, meandiffassistant applies to all tokens in assistant part")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_path", type=str, default='./steer_vectors')
    
    parser.add_argument("--per_cluster_count", type=int, default=100, help="Number of examples per cluster")
    parser.add_argument("--n_clusters", type=int, default=1, help="Number of clusters")
    parser.add_argument("--clustering_eps", type=float, default=0.0, help="Epsilon for clustering differential privacy")
    parser.add_argument("--prompt_style", type=str, default='ptz', help="Prompt style to use")

    parser.add_argument("--neg_data_count", type=int, default=1000, help="Number of negative examples per cluster")
    parser.add_argument("--neg_data_temperature", type=float, default=1.6, help="Temperature used for negative data generation")
    
    parser.add_argument("--neg_data_source", type=str, default='Few', help="Source of negative data")
    parser.add_argument("--neg_data_fixed_shots_epsilon", type=float, default=0.1, help="Epsilon used for fixed shots privacy")
    parser.add_argument("--neg_data_drop_threshold", type=int, default=6, help="Drop threshold used when generating negative data")
    
    parser.add_argument("--n_fixed_shots", type=int, default=1, help="Number of fixed shots in the prompt")
    parser.add_argument("--n_data_shots", type=int, default=1, help="Number of data shots in the prompt")
        
    parser.add_argument("--truncation", type=int, default=800, help="Truncation length for the prompt in tokens")

    parser.add_argument("--clip", type=str2bool, default=False, help="Whether to clip the vectors or not")
    parser.add_argument("--normalization", type=str, default='after', choices=['before', 'after'], help="Normalization type, before aggregation or after aggregation")

    parser.add_argument("--vector_type", type=str, default='meandiff', choices=['meandiff'], help="Type of vector to use for the concept; meandiff only applies to last token, meandiffall applies to all tokens, meandiffassistant applies to all tokens in assistant part")


    if run_in_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    args.device = "auto" if torch.cuda.is_available() else "cpu"

    return args

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

args = get_args(run_in_notebook = in_notebook())
print('Extracting steer vectors with args:', args)

# %%
if '70B' in args.model_name or '12b' in args.model_name or '32B' in args.model_name:
    args.use_quantization = True
    print('Automatically turning on quantization...')

if args.neg_data_fixed_shots_epsilon == float('inf'):
    args.neg_data_fixed_shots_delta = 1.0
elif args.neg_data_fixed_shots_epsilon > 0.0:
    args.neg_data_fixed_shots_delta = 1e-6
else:
    args.neg_data_fixed_shots_delta = 0.0
    
if args.clustering_eps == float('inf'):
    args.clustering_delta = 1.0
elif args.clustering_eps > 0.0:
    args.clustering_delta = 1e-6
else:
    args.clustering_delta = 0.0

seed_everywhere(args.seed)

model_short_names_dict = {'meta-llama/Llama-3.2-1B-Instruct': 'Llama3.2_1B_IT',
                          'meta-llama/Llama-3.1-8B-Instruct': 'Llama3.1_8B_IT',
                          'meta-llama/Llama-3.1-70B-Instruct': 'Llama3.1_70B_IT',
                          'google/gemma-3-4b-it': 'Gemma3_4B_IT',
                          'google/gemma-3-12b-it': 'Gemma3_12B_IT',
                          'Qwen/Qwen3-4B-Instruct-2507': 'Qwen3_4B_IT',
                          'Qwen/Qwen3-4B': 'Qwen3_4B',
                          'Qwen/Qwen3-8B': 'Qwen3_8B',
                          'Qwen/Qwen3-4B-Base': 'Qwen3_4B_PT',
                          'google/gemma-2-2b-it': 'Gemma2_2B_IT',
                          'google/gemma-2-2b': 'Gemma2_2B',
                          'meta-llama/Llama-3.2-1B': 'Llama3.2_1B_PT',
                          'meta-llama/Llama-3.1-8B': 'Llama3.1_8B_PT',
                          'meta-llama/Llama-3.1-70B': 'Llama3.1_70B_PT',
                          'google/gemma-3-4b-pt': 'Gemma3_4B_PT',
                          'google/gemma-3-12b-pt': 'Gemma3_12B_PT',
                          'allenai/Olmo-3-7B-Think': 'Olmo3_7B',
                          'allenai/Olmo-3-7B-Instruct': 'Olmo3_7B_IT',
                          'allenai/Olmo-3-32B-Think': 'Olmo3_32B',
                          'allenai/Olmo-3-1025-7B': 'Olmo3_7B_PT'
                        }
model_short_name = model_short_names_dict[args.model_name]                  


from LLMs.my_gemma2 import SteeredGemma2ForCausalLM
from LLMs.my_gemma3 import SteeredGemma3ForCausalLM
from LLMs.my_llama import SteeredLlamaForCausalLM
from LLMs.my_olmo3 import SteeredOlmo3ForCausalLM
from LLMs.my_qwen3 import SteeredQwen3ForCausalLM
from transformers import BitsAndBytesConfig


model_classes_dict = {    'meta-llama/Llama-3.2-1B-Instruct': SteeredLlamaForCausalLM,
                          'meta-llama/Llama-3.1-8B-Instruct': SteeredLlamaForCausalLM,
                          'meta-llama/Llama-3.1-70B-Instruct': SteeredLlamaForCausalLM,
                          'google/gemma-3-4b-it': SteeredGemma3ForCausalLM,
                          'google/gemma-3-12b-it': SteeredGemma3ForCausalLM,
                          'Qwen/Qwen3-4B-Instruct-2507': SteeredQwen3ForCausalLM,
                          'Qwen/Qwen3-4B': SteeredQwen3ForCausalLM,
                          'Qwen/Qwen3-8B': SteeredQwen3ForCausalLM,
                          'Qwen/Qwen3-4B-Base': SteeredQwen3ForCausalLM,
                          'google/gemma-2-2b-it': SteeredGemma2ForCausalLM,
                          'google/gemma-2-2b': SteeredGemma2ForCausalLM,
                          'meta-llama/Llama-3.2-1B': SteeredLlamaForCausalLM,
                          'meta-llama/Llama-3.1-8B': SteeredLlamaForCausalLM,
                          'meta-llama/Llama-3.1-70B': SteeredLlamaForCausalLM,
                          'google/gemma-3-4b-pt': SteeredGemma3ForCausalLM,
                          'google/gemma-3-12b-pt': SteeredGemma3ForCausalLM,
                          'allenai/Olmo-3-7B-Think': SteeredOlmo3ForCausalLM,
                          'allenai/Olmo-3-7B-Instruct': SteeredOlmo3ForCausalLM,
                          'allenai/Olmo-3-32B-Think': SteeredOlmo3ForCausalLM,
                          'allenai/Olmo-3-1025-7B': SteeredOlmo3ForCausalLM
                        }

model_class = model_classes_dict[args.model_name]          

if 'gemma' in args.model_name.lower():
    assistant_tag = '<start_of_turn>model\n'
elif 'llama' in args.model_name.lower():
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>\n\n'    
elif 'qwen' in args.model_name.lower():
    assistant_tag = '<|im_start|>assistant\n'
elif 'olmo' in args.model_name.lower():
    assistant_tag = '<|im_start|>assistant\n'
else:
    raise ValueError(f"Model {args.model_name} is not supported")

tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if args.use_quantization:    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_short_name = model_short_name + "_quantized"
else:
    quantization_config = None

model = model_class.from_pretrained(args.model_name, device_map=args.device, quantization_config=quantization_config)
model.eval()

model_short_name_with_temperature = f"{model_short_name}_temperature_{args.neg_data_temperature}"

data_raw, class_labels_to_ids = load_binary_pairs(dataset_name = args.dataset,
                    per_cluster_count = args.per_cluster_count,
                    prompt_style = args.prompt_style,
                    neg_data_model_short_name_with_temperature = model_short_name_with_temperature,
                    neg_data_source = args.neg_data_source,
                    neg_data_seed = args.seed,
                    neg_data_count = args.neg_data_count,
                    n_fixed_shots = args.n_fixed_shots,
                    n_data_shots = args.n_data_shots,
                    seed = args.seed,
                    truncation = args.truncation,
                    tokenizer = tokenizer,
                    neg_data_drop_threshold = args.neg_data_drop_threshold,
                    neg_data_fixed_shots_epsilon = args.neg_data_fixed_shots_epsilon,
                    neg_data_fixed_shots_delta = args.neg_data_fixed_shots_delta,
                    clustering_eps = args.clustering_eps,
                    clustering_delta = args.clustering_delta,
                    n_clusters = args.n_clusters
                    )



if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    print("The model has a chat template configured.")
    # print(f"Chat template: {tokenizer.chat_template}")
    chat_enabled = True
else:
    chat_enabled = False
    print("The model does not have an explicit chat template configured.")

if chat_enabled:
    messages_with_system = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
        ]

    try:
        # Attempt to apply the template with a system message
        
        formatted_input = tokenizer.apply_chat_template(messages_with_system, tokenize=False)
        print("System role is likely supported.")
        system_role_supported = True
        # You can also inspect `tokenizer.chat_template` if it's explicitly defined
        # print(tokenizer.chat_template)
    except Exception as e:
        
        if "System role not supported" in str(e):
            print("System role is not supported by this model's tokenizer.")
            system_role_supported = False
        else:
            print(f"An error occurred: {e}")

def apply_chat_template(data, tokenizer):
    all_data = []
    for d in data:
        user_prompt = d.get('user', '')
        system_prompt = d.get('system', '')
        assistant_prompt = d.get('assistant', '')
        chat = []
        if chat_enabled:
            if system_role_supported:
                if system_prompt != '':
                    chat.append({'role': 'system', 'content': system_prompt})

                chat.append({'role': 'user', 'content': user_prompt})
            else:
                system_ = f'{system_prompt}\n\n' if system_prompt != '' else ''
                chat.append({'role': 'user', 'content': system_ + user_prompt})

            chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking = False)

            if 'assistant' in d:
                chat += d['assistant']

            if tokenizer.bos_token:
                chat = chat.replace(tokenizer.bos_token, '')
        else:

            if system_prompt != '':
                user_prompt = system_prompt + '\n\n'+ user_prompt
            if assistant_prompt != '':
                user_prompt = user_prompt + '\n\n' + assistant_prompt
            chat = user_prompt

        all_data.append(chat)
    return all_data

def recursively_apply_chat_template(data_dict, tokenizer):
    new_data_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            new_data_dict[key] = recursively_apply_chat_template(value, tokenizer)
        elif isinstance(value, list):
            new_data_dict[key] = apply_chat_template(value, tokenizer)
        else:
            new_data_dict[key] = value
    return new_data_dict

data = recursively_apply_chat_template(data_raw, tokenizer)
print(data['pos'].keys())

path = args.save_path + f'/{model_short_name_with_temperature}/Seed_{args.seed}/{args.dataset}/{args.prompt_style}/c_{args.n_clusters}_perc_{args.per_cluster_count}_rce_{args.clustering_eps}_rcd_{args.clustering_delta}_negc_{args.neg_data_count}/sc_{args.neg_data_source}_fs_{args.n_fixed_shots}_shots_{args.n_data_shots}_eps_{args.neg_data_fixed_shots_epsilon}_delta_{args.neg_data_fixed_shots_delta}_dt_{args.neg_data_drop_threshold}/{args.target_tokens}/'
print('Path:', path)
os.makedirs(path, exist_ok=True)

# %%
extraction_locs = [7]
extraction_layers = list(range(model.config.num_hidden_layers))

extraction_tokens = [-1]
if 'assistant' in args.target_tokens:
    extraction_tokens = 'assistant'
elif 'all' in args.target_tokens:
    extraction_tokens = 'all'
else:
    extraction_tokens = [-1]

def get_estimate_of_total_data(data_dict):
    total = 0
    for key in data_dict.keys():
        if isinstance(data_dict[key], dict):
            total += get_estimate_of_total_data(data_dict[key])
        elif isinstance(data_dict[key], list):
            total += len(data_dict[key])
        else:
            raise ValueError("Data structure not recognized")
    return total


def get_estimate_of_lengths(data_dict, structure=''):
    token_lengths = {}
    word_lengths = {}
    for key in data_dict.keys():
        if isinstance(data_dict[key], dict):
            structure_ = structure + f'/{key}'
            token_lengths[key], word_lengths[key] = get_estimate_of_lengths(data_dict[key], structure=structure_)
        elif isinstance(data_dict[key], list):
            token_lengths[key] = []
            word_lengths[key] = []
            for d in data_dict[key]:
                tokenized = tokenizer(d, return_tensors='pt', truncation=False)
                token_lengths[key].append(tokenized['input_ids'].shape[1])
                word_lengths[key].append(len(d.split()))

            token_lengths[key] = sorted(token_lengths[key], reverse=True)
            word_lengths[key] = sorted(word_lengths[key], reverse=True)
            print(f"Estimated lengths at {structure}/{key}:")
            print(f"  Token lengths (top 10): {token_lengths[key][:10]}")
            print(f"  Word lengths (top 10): {word_lengths[key][:10]}")
            print('---------------------------------------------------------')
        else:
            raise ValueError("Data structure not recognized")
    return token_lengths, word_lengths


print(f"Estimated total data size: {get_estimate_of_total_data(data)}")
# print("Estimating lengths of data samples...")
# token_lengths, word_lengths = get_estimate_of_lengths(data, structure='')
# exit()
def recursively_embed_data(data_dict, structure=''):
    embedded_data = {}
    for key in data_dict.keys():
        if isinstance(data_dict[key], dict):
            structure_ = structure + f'/{key}'
            embedded_data[key] = recursively_embed_data(data_dict[key], structure=structure_)
        elif isinstance(data_dict[key], list):
            print(f"Embedding data at {structure}/{key} with {len(data_dict[key])} examples...")
            dataset = TextDataset(data_dict[key])
            dataloader = DataLoader(dataset, batch_size = args.bs, shuffle=False)
            try:
                all_hidden_states = extract_hidden_states(dataloader, tokenizer, model, assistant_tag, extraction_locs=extraction_locs, extraction_layers=extraction_layers, extraction_tokens = extraction_tokens, do_final_cat=True, avg_token_dim=True)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    # find the maximum length in the dataset
                    lens = []
                    for d in data_dict[key]:
                        tokenized = tokenizer(d, return_tensors='pt', truncation=False)
                        lens.append(tokenized['input_ids'].shape[1])
                    print('Sorted lengths:', sorted(lens, reverse=True)[:100])
                    torch.cuda.empty_cache()
                    raise RuntimeError(f"CUDA out of memory when processing data at {structure}/{key} with maximum length {max(lens)}. Consider reducing batch size or truncating data.")
                else:
                    raise e
            embedded_data[key] = all_hidden_states
        else:
            raise ValueError("Data structure not recognized")
    return embedded_data


dataset = TextDataset([d for t in data for d in t])
print(f"Dataset size: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size = args.bs, shuffle=False)


if not os.path.exists(path + 'embedded_data.pt'):
    embedded_data = recursively_embed_data(data)
    torch.save(embedded_data, path + 'embedded_data.pt')
else:
    print('Loading existing embedded data...')
    embedded_data = torch.load(path + 'embedded_data.pt')



def clip_in_norm(vecs, clip_value, norm_type=2):
    norms = torch.norm(vecs, p=norm_type, dim=-1, keepdim=True)
    scaling_factors = torch.clamp(norms, max=clip_value) / (norms + 1e-10)
    return vecs * scaling_factors

coeffs = torch.tensor(load_json(f'methods/hyperparams/{model_short_name}.json')['clip'], dtype=torch.float32)
coeffs_expanded = coeffs.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Expand dimensions for broadcasting
if args.vector_type == 'meandiff':
    embeddings_diff = {}

    for cls in embedded_data['pos'].keys():
        embeddings_diff[cls] = {}
        for cluster in embedded_data['pos'][cls].keys():
            d = embedded_data['pos'][cls][cluster] - embedded_data['neg'][cls]
    
            if args.clip:
                d_bar = clip_in_norm(d / coeffs_expanded, clip_value=1.0, norm_type=2)
                if args.normalization == 'after':
                    d_bar = d_bar * coeffs_expanded
                    
                v = d_bar.mean(dim=0)

            else:
                if args.normalization == 'before':
                    d = d / torch.norm(d, p=2, dim=-1, keepdim=True)
                
                v = d.mean(dim=0)
            
            embeddings_diff[cls][cluster] = v
else:
    raise ValueError(f"Vector type {args.vector_type} not supported")    


for cls in embeddings_diff.keys():
    for cluster in embeddings_diff[cls].keys():
        for l, layer in enumerate(extraction_layers):
            for lc, loc in enumerate(extraction_locs):
                save_dir = path + f'/vt_{args.vector_type}_norm_{args.normalization}_clip_{args.clip}/{cls}/{cluster}/layer_{layer}/loc_{loc}/'
                os.makedirs(save_dir, exist_ok=True)
                torch.save(embeddings_diff[cls][cluster][l][lc], save_dir + 'steer_vector.pt')