# %%
# 


# %%
import argparse
from tqdm.auto import tqdm
import os
import random
# from path_manager import get_main_path
import gc
from my_utils import str2bool
from my_utils import load_json, save_json

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# %%


def get_args(run_in_notebook=False):
    parser = argparse.ArgumentParser(description='Run the LLM')

    # Model arguments
    parser.add_argument("--model_name", type=str, help="Model Name", default="meta-llama/Llama-3.1-8B")
    
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation')
    parser.add_argument("--use_quantization", type=str2bool, default=False, help="Whether to use quantization for the model")
    parser.add_argument('--dataset', type=str, default='yelp', help='Evaluation dataset to use')
    

    parser.add_argument('--result_save_path', type=str, default='results/', help='Path to save the results')
    parser.add_argument('--skip_with_file', type=str, default=None)
    # Inference arguments
    parser.add_argument('--bs', type=int, default=3, help='Physical batch size limit by GPU memory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=1.6, help='Temperature for sampling')
    parser.add_argument('--argmax', type=str2bool, default=False, help='Whether to use argmax sampling instead of temperature sampling')
    parser.add_argument('--use_kv_cache', type=str2bool, default=True, help='Whether to use kv cache during generation')
    parser.add_argument('--eval_only', type=str2bool, default=False, help='Whether to only evaluate the existing results')
    parser.add_argument('--top-p', type=float, default=None, help='Top-p sampling parameter')
    parser.add_argument('--top-k', type=int, default=None, help='Top-k sampling parameter')

    parser.add_argument('--method', type=str, default='Few', choices=['VI', 'CVI', 'Few'], help='Method to use for generation; use Few for 2-shot and 2-fixed-shot experiments and VI for EPSVec.')

    # General Parameters
    parser.add_argument('--prompt_style', type=str, default='ptz')
    parser.add_argument('--count', type=int, default=1000, help='Number of samples to generate per class')
    
    parser.add_argument('--drop_threshold', type=int, default=6, help='Drop samples below this quality threshold')
    parser.add_argument('--pp_improve', type=str2bool, default=False, help='Whether to use an IT model to improve quality of generations')
    
    # Few-shot specific parameters
    parser.add_argument('--n_fixed_shots', type=int, default=2, help='Number of fixed shots for few-shot learning')
    parser.add_argument('--n_shots', type=int, default=0, help='Number of shots for few-shot learning')
    parser.add_argument('--fixed_shots_epsilon', type=float, default=0.1, help='Epsilon value for fixed shots in few-shot learning')
    
    # All private methods specific parameters
    parser.add_argument('--epsilon', type=float, default='inf', help='Epsilon value for private methods')
    
    # Vector Injection specific parameters (inherited from Few-shot)
    
    parser.add_argument("--vec_target_tokens", type=str, default='all', choices=['last', 'all', 'assistant'], help="Tokens used to extract steering vectors")
    parser.add_argument("--vec_per_cluster_count", type=int, default=500, help="Number of examples used per cluster for vector extraction")
    parser.add_argument("--vec_normalization", type=str, default='after', choices=['before', 'after'], help="Normalization type, applied before or after averaging")
    parser.add_argument('--vec_type', type=str, default='meandiff', choices=['meandiff', 'probe'], help='Type of vector to use for injection')
    parser.add_argument('--vec_temp', type=float, default=None, help='Temperature used for generating data for vector extraction')
    parser.add_argument('--vec_drop_threshold', type=int, default=6, help='Quality drop threshold for generating data for vector extraction')
    
    parser.add_argument('--injection_reweighting', type=str2bool, default=False, help='Whether to reweight the injection based on layer norms')
    parser.add_argument('--injection_coeff', type=float, default=1.4, help='Coefficient for vector injection')
    parser.add_argument('--injection_layers', type=str, default='18,19,20,21', help='Layers to inject the vector into, e.g., all or 0,1,2')
    parser.add_argument('--injection_location', type=str, default='7', help='"7" means injection to hidden states')
    
    parser.add_argument('--noise_scale', type=float, default=0, help='Scale of noise to add during vector injection')
    parser.add_argument('--noise_layers', type=str, default='all', help='Layers to which noise is added during vector injection, e.g., "all", "15,16", or "injection')
    
    # CVI specific parameters
    parser.add_argument('--vec_n_clusters', type=int, default=1, help='Number of clusters to use for CVI')
    parser.add_argument("--vec_clustering_epsilon", type=float, default=0.0, help="Epsilon value for clustering in CVI")
    
    parser.add_argument('--resume', type=str2bool, default=False, help='Whether to resume or start fresh')
    
    if run_in_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
        
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

# %%

if '70B' in args.model_name or '12b' in args.model_name or '32B' in args.model_name:
    args.use_quantization = True
    print('Automatically turning on quantization...')

# %%
if args.argmax:
    print("Using argmax sampling, setting temperature to 0.6")
    args.temperature = 0.6

if args.vec_temp is None:
    args.vec_temp = args.temperature

if args.epsilon != float('inf'):
    print(f'Setting delta to 1e-5 for finite epsilon')
    args.delta = 1e-5
else:
    print(f'Setting delta to 1.0 for infinite epsilon')
    args.delta = 1.0

if args.fixed_shots_epsilon == float('inf'):
    print(f'Setting fixed shots delta to 1.0 for infinite epsilon')
    args.fixed_shots_delta = 1.0
elif args.fixed_shots_epsilon == 0.0:
    print(f'Setting fixed shots delta to 0.0 for zero epsilon')
    args.fixed_shots_delta = 0.0
else:
    print(f'Setting fixed shots delta to 1e-6 for finite epsilon')
    args.fixed_shots_delta = 1e-6

if args.vec_clustering_epsilon == float('inf'):
    print(f'Setting clustering delta to 1.0 for infinite epsilon')
    args.vec_clustering_delta = 1.0
elif args.vec_clustering_epsilon == 0.0:
    print(f'Setting clustering delta to 0.0 for zero epsilon')
    args.vec_clustering_delta = 0.0
else:
    print(f'Setting clustering delta to 1e-6 for finite epsilon')
    args.vec_clustering_delta = 1e-6

# %%
print(f'Model name: {args.model_name}')
print(f'Using quantization: {args.use_quantization}')

print(f'Dataset: {args.dataset}')
print(f'Batch size: {args.bs}')
print(f'Argmax or Sample: {"argmax" if args.argmax else "sample"}')
print(f'Temperature:', args.temperature)
print(f'Method: {args.method}')
print(f'Prompt style: {args.prompt_style}')
print(f'Count per class: {args.count}')

if args.method in ['Few']:
    print(f'Number of fixed shots: {args.n_fixed_shots}, shots: {args.n_shots}, fixed shots epsilon: {args.fixed_shots_epsilon}, fixed shots delta: {args.fixed_shots_delta}')
elif 'VI' in args.method:
    print(f'Number fixed shots: {args.n_fixed_shots}, fixed shots epsilon: {args.fixed_shots_epsilon}, fixed shots delta: {args.fixed_shots_delta}')
    print(f'Vector target tokens: {args.vec_target_tokens}, per cluster count: {args.vec_per_cluster_count}, normalization: {args.vec_normalization}, type: {args.vec_type}')
    print(f'Injection reweighting: {args.injection_reweighting}, injection coefficient: {args.injection_coeff}, injection layers: {args.injection_layers}, injection location: {args.injection_location}')
    if 'CVI' in args.method:
        print(f'Number of clusters: {args.vec_n_clusters}, clustering epsilon: {args.vec_clustering_epsilon}, clustering delta: {args.vec_clustering_delta}')
    print(f'Epsilon: {args.epsilon}, Delta: {args.delta}')

# %%

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

print('model_name:', args.model_name)

if args.use_quantization:
    model_short_name = model_short_name + "_quantized"




# %%
from methods.Few import Few
from methods.VI import VI
from methods.CVI import CVI

baseline: Few = None

if args.method == 'Few':
    baseline = Few(dataset=args.dataset, prompt_style=args.prompt_style, count_per_class=args.count, bs=args.bs,
                   n_shots=args.n_shots, n_fixed_shots=args.n_fixed_shots, fixed_shots_epsilon=args.fixed_shots_epsilon, fixed_shots_delta=args.fixed_shots_delta,
                   eval_only=args.eval_only, seed=args.seed, quality_threshold=args.drop_threshold, improve_quality=args.pp_improve)
elif args.method == 'VI':
    baseline = VI(dataset=args.dataset, prompt_style=args.prompt_style, count_per_class=args.count, bs=args.bs,
                  n_fixed_shots=args.n_fixed_shots, fixed_shots_epsilon=args.fixed_shots_epsilon, fixed_shots_delta=args.fixed_shots_delta, vec_target_tokens=args.vec_target_tokens, vec_per_cluster_count=args.vec_per_cluster_count,
                  vec_normalization=args.vec_normalization, vec_type=args.vec_type, vec_temp=args.vec_temp, vec_drop_threshold=args.vec_drop_threshold,
                  injection_reweighting=args.injection_reweighting, injection_coeff=args.injection_coeff,
                  injection_layers=args.injection_layers, injection_location=args.injection_location,
                  model_short_name=model_short_name, eval_only=args.eval_only, seed=args.seed,
                  epsilon=args.epsilon, delta=args.delta, quality_threshold=args.drop_threshold, improve_quality=args.pp_improve,
                  noise_scale=args.noise_scale, noise_layers=args.noise_layers)
elif args.method == 'CVI':
    baseline = CVI(dataset=args.dataset, prompt_style=args.prompt_style, count_per_class=args.count, bs=args.bs,
                   n_fixed_shots=args.n_fixed_shots, fixed_shots_epsilon=args.fixed_shots_epsilon, fixed_shots_delta=args.fixed_shots_delta, vec_target_tokens=args.vec_target_tokens, vec_per_cluster_count=args.vec_per_cluster_count,
                   vec_normalization=args.vec_normalization, vec_type=args.vec_type, vec_temp=args.vec_temp, vec_drop_threshold=args.vec_drop_threshold,
                   injection_reweighting=args.injection_reweighting, injection_coeff=args.injection_coeff,
                   injection_layers=args.injection_layers, injection_location=args.injection_location,
                   model_short_name=model_short_name, eval_only=args.eval_only, seed=args.seed,
                   n_clusters=args.vec_n_clusters, clustering_eps=args.vec_clustering_epsilon, clustering_delta=args.vec_clustering_delta,
                   epsilon=args.epsilon, delta=args.delta, quality_threshold=args.drop_threshold, improve_quality=args.pp_improve,
                   noise_scale=args.noise_scale, noise_layers=args.noise_layers)
else:
    raise NotImplementedError(f'Method {args.method} not implemented yet.')

temperature_str = '' if args.argmax else f'_temperature_{args.temperature}'

save_dir = f'{args.result_save_path}/{model_short_name}{temperature_str}/Seed_{args.seed}/{baseline.get_path_name()}'


# %%
if args.skip_with_file:
    file_path = save_dir + '/' + args.skip_with_file
    if os.path.exists(file_path):
        print('File path exists, exiting the run:', file_path)
        assert False, 'Exit...'
if not args.eval_only:
    os.makedirs(save_dir, exist_ok=True)
print('created directory:', save_dir)

# %%
import torch
import numpy as np
from transformers import AutoTokenizer, BitsAndBytesConfig, StoppingCriteria
# from steer_manager import get_steer_fn, load_steer_vectors, apply_wieghted_sum, get_default_steer_fn
import torch.nn.functional as F
def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed_everywhere(args.seed)

if args.eval_only:
    print('Evaluation only mode, skipping model loading...')
    baseline.finalize(save_path=save_dir)
    exit()
    
    

# %%

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

print('model:', args.model_name, model_class, model_short_name)

# %%
if args.use_quantization:    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
else:
    quantization_config = None


def load_model():
    
    print('Loading model for data generation...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token    

    
    model = model_class.from_pretrained(args.model_name, device_map=args.device, quantization_config=quantization_config) 
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    torch.cuda.empty_cache()
    model, tokenizer = baseline.prepare_inference(model, tokenizer)
    return model, tokenizer

model, tokenizer = load_model()

# %%


if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    print("The model has a chat template configured.")
    # print(f"Chat template: {tokenizer.chat_template}")
    chat_enabled = True
else:
    chat_enabled = False
    print("The model does not have an explicit chat template configured.")

# %%
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



def get_stopping_criteria(stop_strings: list[str], tokenizer, original_token_lens):
    if stop_strings is None:
        return None
    class StopOnString(StoppingCriteria):
        def __init__(self, stop_strings: list[str], tokenizer, original_token_lens: list):
            self.stop_strings = stop_strings
            self.tokenizer = tokenizer
            self.original_token_lens = original_token_lens
            # self.stop_token_ids = tokenizer.encode(stop_string, add_special_tokens=False)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            
            for i in range(input_ids.shape[0]):
                generated_tokens = input_ids[i, self.original_token_lens[i]:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                if not any(stop_string in generated_text for stop_string in self.stop_strings):
                    return False
            return True
            
            

    return [StopOnString(stop_strings, tokenizer, original_token_lens)]
# %%

last_progress = 0
max_tokenized_len = 0
counter = 0


if args.resume:
    print('[RESUME] Resuming from previous run...')
    RESUME_FILE_NAME = 'progress.json'
    
    if os.path.exists(os.path.join(save_dir, RESUME_FILE_NAME)):
        loading_finished = False
        saveable_json = load_json(os.path.join(save_dir, RESUME_FILE_NAME))
        print(f'[RESUME] Loaded progress file with {len(saveable_json)} entries.')
        max_saved_counter = max([int(k) for k in saveable_json.keys()])
    else:
        print('[RESUME] No progress file found, starting fresh...')
        loading_finished = True
        saveable_json = {}


with tqdm(total=100, unit="iteration", desc=f"Running method {baseline.get_unique_name()} on {args.dataset}") as pbar:
    while not baseline.is_finished():
        model, tokenizer = baseline.modify_language_model(model, tokenizer)
        
        user_prompts = baseline.get_user_prompt()
        system_prompts = baseline.get_system_prompt()
        assistant_prompts = baseline.get_assistant_prompt()

        if not isinstance(user_prompts, list):
            user_prompts = [user_prompts]
        
        if not isinstance(system_prompts, list):
            system_prompts = [system_prompts]
        
        if not isinstance(assistant_prompts, list):
            assistant_prompts = [assistant_prompts]
        
        assert len(user_prompts) == len(system_prompts) == len(assistant_prompts), f"User prompts: {len(user_prompts)}, System prompts: {len(system_prompts)}, Assistant prompts: {len(assistant_prompts)}"
            
        
        new_prompts = []
        for user_prompt, system_prompt, assistant_prompt in zip(user_prompts, system_prompts, assistant_prompts):

            if chat_enabled:
                chat = []
                if system_role_supported and system_prompt != '':
                    chat.append({'role': 'system', 'content': system_prompt})
                else:
                    user_prompt = system_prompt + user_prompt

                chat.append({'role': 'user', 'content': user_prompt})
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False) + assistant_prompt

                # removing bos token from the prompt because it is added by the tokenizer again in the future tokenize call
                bos_token = tokenizer.bos_token
                if tokenizer.bos_token:
                    prompt = prompt.replace(tokenizer.bos_token, '')
            else:
                if system_prompt != '':
                    user_prompt = system_prompt + '\n\n'+ user_prompt
                if assistant_prompt != '':
                    user_prompt = user_prompt + '\n\n' + assistant_prompt
                prompt = user_prompt
            new_prompts.append(prompt)
        
        prompts = new_prompts

            
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
        len_original_prompts = [len(x) for x in inputs.input_ids]
        
        generation_kwargs = dict(
            max_new_tokens=baseline.get_max_len(),
            do_sample=not args.argmax,
            temperature=args.temperature,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=args.use_kv_cache,
            disable_compile=True,
            stopping_criteria=get_stopping_criteria(
                baseline.get_stopping_criteria(), tokenizer, len_original_prompts
            ),
        )

        # If top-p or top-k are specified, add them to generation kwargs. 
        for name, value in {"top_p": args.top_p, "top_k": args.top_k}.items():
            if value is not None:
                generation_kwargs[name] = value
        
        
        if args.resume and not loading_finished:
            cropped_generations, full_response = saveable_json[str(counter)]
            print(f'[RESUME] Skipping already processed example at step {counter}.')
            if counter == max_saved_counter:
                loading_finished = True
                print('[RESUME] Finished loading previous progress, resuming normal generation...')
        else:
            with torch.no_grad():
                generation = model.generate(**inputs, **generation_kwargs)

            cropped_generations = []
            full_response = []
            for b in range(inputs.input_ids.shape[0]):
                cropped_generations.append(tokenizer.decode(generation.sequences[b, len_original_prompts[b]:], skip_special_tokens=True))
                full_response.append(tokenizer.decode(generation.sequences[b, :], skip_special_tokens=False))

            del generation
            gc.collect()
            torch.cuda.empty_cache()
            
                
            
        
        if (counter + 1) % 100 == 0:
            print(f'Example generations at step {counter}:')
            print(f'Prompt: {prompts[0]}')
            print('----------------------------------------')
            print(f'Generation: {cropped_generations[0]}')
            print('----------------------------------------')
        
        if args.resume and loading_finished:
            full_response_ = full_response if counter == 0 else '-'
            
            if args.method in ['PP', 'PPClustering']:
                cropped_generations = cropped_generations[:1]
            
            saveable_json[str(counter)] = (cropped_generations, full_response_)
            if (counter % 10 == 0):
                save_json(saveable_json, os.path.join(save_dir, RESUME_FILE_NAME))
                print(f'[RESUME] Saved progress at step {counter}...')
            
        
        counter += 1
                
        baseline.process_results(llm_generations=cropped_generations, full_prompt=full_response)
        torch.cuda.empty_cache()
        
        pbar.update(100 * (baseline.get_progress() - last_progress))
        last_progress = baseline.get_progress()

    pbar.update((baseline.get_progress() - last_progress) == 1.0)

if baseline.clean_gpu_memory():
    try:
        baseline.clean_cache()
        del model, inputs, tokenizer
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()

baseline.finalize(save_path=save_dir)
