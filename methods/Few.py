import os
from methods.PromptGenerator import PromptGenerator
from dataset_loader import classes_to_labels, datasets_to_functions
from my_utils import save_json, load_json
from tqdm.auto import tqdm
import gc

import torch  
import numpy as np
from my_mauve import MauveEvaluator
import quality_evaluator as qe

from icl_evaluator import ICLEvaluator
from BERT_Evaluator import BERTEvaluator
from quality_enhancer import QualityImprover


class Few:
    def __init__(self, eval_only, dataset, prompt_style, count_per_class, bs, n_shots, n_fixed_shots, fixed_shots_epsilon, fixed_shots_delta, seed, quality_threshold, improve_quality):
        
        self.prompt_style = prompt_style
        self.count_per_class = count_per_class
        self.bs = bs
        self.n_shots = n_shots
        self.n_fixed_shots = n_fixed_shots
        self.seed = seed
        self.quality_threshold = quality_threshold
        assert 0 <= quality_threshold <= 10, "quality_threshold must be between 0 and 10."
        self.improve_quality = improve_quality # improve quality in a post-processing step
        
        self.fixed_shots_epsilon = fixed_shots_epsilon
        self.fixed_shots_delta = fixed_shots_delta
        
        if quality_threshold > 0:
            self.fast_quality_checker = qe.QualityEvaluator('Qwen/Qwen3-4B-Instruct-2507', quantization=True, thinking_budget=0)
            self.total_waste_budget = 3 * count_per_class
        
        self.dataset = dataset
        
        self.class_labels_to_id = classes_to_labels[dataset.lower()]
        self.class_labels = list(self.class_labels_to_id.keys())
        self.ids_to_class_labels = {v: k for k, v in self.class_labels_to_id.items()}

        self.prompt_managers = {}
        for cls in self.class_labels:
            self.prompt_managers[cls] = PromptGenerator(dataset, cls, prompt_style, n_shots=n_shots, n_fixed_shots=n_fixed_shots, fixed_shots_epsilon=fixed_shots_epsilon, fixed_shots_delta=fixed_shots_delta)

        self.few_shot_set = datasets_to_functions[dataset.lower()](size=n_shots * count_per_class, seed=seed, keys=['train'])['train']
        self.total_length = len(self.class_labels) * count_per_class
        
        self.counter = 0
        self.in_class_counter = 0
        self.current_class_index = 0
        
        self.eval_only = eval_only
        
        self.sample_full_prompt = {}
        for cls in self.class_labels:
            self.sample_full_prompt[cls] = None
        self.sample_s = None
        self.sample_u = None
        self.sample_a = None
        
        self.results = []
        
        self.compare_set = datasets_to_functions[self.dataset.lower()](size=self.count_per_class, seed=seed)
        
        assert self.n_fixed_shots > 0 or self.fixed_shots_epsilon == 0.0, "If no fixed shots are used, fixed_shots_epsilon must be 0.0"
        

    def get_shared_path_name(self):
        return f'{self.dataset}/{self.prompt_style}/count_{self.count_per_class}_qt_{self.quality_threshold}/'
    
    def get_path_name(self):
        epsilon, delta = self.get_prompt_template_epsilon_delta()
        return self.get_shared_path_name() + f'Few_{self.n_fixed_shots}_{self.n_shots}/eps_{epsilon}_delta_{delta}/'
    
    def get_batch_size(self):
        return self.bs
    
    def clean_gpu_memory(self):
        return True
    
    def get_preferred_batch_size(self):
        preferred_batch_size = self.get_batch_size()
        bs = min(preferred_batch_size, self.count_per_class - self.in_class_counter)
        return bs

    def get_system_prompt_(self):
        current_class_prompt_manager = self.prompt_managers[self.class_labels[self.current_class_index]]
        return current_class_prompt_manager.get_system_prompt(self.get_preferred_batch_size())

    def get_user_prompt_(self):
        pfbs = self.get_preferred_batch_size()
        current_class_prompt_manager: PromptGenerator = self.prompt_managers[self.class_labels[self.current_class_index]]
        few_shot_samples = self.few_shot_set[self.current_class_index][self.in_class_counter * self.n_shots:self.in_class_counter * self.n_shots + pfbs * self.n_shots]
        few_shot_samples = current_class_prompt_manager.get_user_prompt(pfbs, few_shot_samples)
        return few_shot_samples


    def get_assistant_prompt_(self):
        current_class_prompt_manager: PromptGenerator = self.prompt_managers[self.class_labels[self.current_class_index]]
        return current_class_prompt_manager.get_assistant_prompt(self.get_preferred_batch_size())

    def get_system_prompt(self):
        prompt_s = self.get_system_prompt_()
        if self.sample_s is None:
            self.sample_s = prompt_s[0]
        return prompt_s
    
    def get_user_prompt(self):
        prompt_u = self.get_user_prompt_()
        if self.sample_u is None:
            self.sample_u = prompt_u[0]
        return prompt_u

    def get_assistant_prompt(self):
        prompt_a = self.get_assistant_prompt_()
        if self.sample_a is None:
            self.sample_a = prompt_a[0]
        return prompt_a

    def is_finished(self):
        return self.counter >= self.total_length or self.eval_only
    
    def process_results(self, llm_generations, full_prompt):
        if self.sample_full_prompt[self.class_labels[self.current_class_index]] is None:
            self.sample_full_prompt[self.class_labels[self.current_class_index]] = full_prompt[0]

        prompt_manager: PromptGenerator = self.prompt_managers[self.class_labels[self.current_class_index]]
        
        post_processed_responses = []
        for response in llm_generations:
            response_ = prompt_manager.post_process_response(response)
            post_processed_responses.append(response_)
        
        if self.quality_threshold > 0:
            quality_scores = self.fast_quality_checker.evaluate(post_processed_responses, bs=8, verbose=False)
            
        
        number_of_accepted = 0
        for i in range(len(post_processed_responses)):
            response_ = post_processed_responses[i]
            fast_quality_score = -1.0
            if self.quality_threshold > 0:
                quality_score = quality_scores[i]['overall']
                fast_quality_score = quality_score
                if quality_score < self.quality_threshold:
                    self.total_waste_budget -= 1
                    # if self.total_waste_budget <= 0:
                    #     raise Exception('Exceeded total waste budget due to low-quality generations.')
                    if self.total_waste_budget > 0:
                        continue
                    
                
            
            self.results.append({'text': response_, 'class': self.class_labels[self.current_class_index], 'raw_text': llm_generations[i], 'fast_quality_score': fast_quality_score})
            number_of_accepted += 1
        
        self.update_counters(number_of_accepted)
        
    
    
    def update_counters(self, len_responses):        
        self.counter += len_responses
        self.in_class_counter += len_responses
        if self.in_class_counter >= self.count_per_class:
            self.in_class_counter = 0
            self.current_class_index = self.current_class_index + 1
        
    def categorize_results_texts_by_class_id(self, results, key):
        categorized = {self.class_labels_to_id[cls]: [] for cls in self.class_labels}        
        for res in results:
            cls = self.class_labels_to_id[res['class']]
            categorized[cls].append(res[key])
        return categorized    
    
    
    def convert_V1_to_V2(self, summary_path_V1, summary_path_V2):
            # Load V1 summary
            from copy import deepcopy
            summary_V1 = load_json(summary_path_V1)
            summary_V2 = deepcopy(summary_V1)
            
            # We just to rerun the BERT evaluation part
            print('Converting V1 summary to V2 by re-evaluating BERT metrics...')
            synthetic_data = {}
            keys = ['text', 'polished'] if self.improve_quality else ['text']
            self.results = summary_V1['details']
            for key in keys:
                synthetic_data[key] = self.categorize_results_texts_by_class_id(self.results, key)
            
            bert_evaluator = BERTEvaluator(self.compare_set['train'], self.compare_set['test'], epochs=5)
            bert_results = {}
            for key in keys:
                bert_results[key] = {}
                for repeat in range(10):
                    bert_results[key][repeat] = bert_evaluator.evaluate(synthetic_data[key], seed=self.seed * 100 + repeat)
            summary_V2['BERT'] = {key: bert_results[key] for key in keys}
            
            save_json(summary_V2, summary_path_V2)
            

    def convert_V2_to_V3(self, summary_path_V2, summary_path_V3, empty_summary_dict):
            # Load V2 summary
            from copy import deepcopy
            summary_V3 = deepcopy(empty_summary_dict)
            summary_V2 = load_json(summary_path_V2)    
            for k in summary_V3.keys():
                if k in summary_V2:
                    summary_V3[k] = deepcopy(summary_V2[k])
            
            
            # We just to rerun the Mauve total evaluation part
            print('Converting V2 summary to V3 by re-evaluating Mauve total metrics...')
            synthetic_data = {}
            keys = ['text', 'polished'] if self.improve_quality else ['text']
            self.results = summary_V2['details']
            for key in keys:
                synthetic_data[key] = self.categorize_results_texts_by_class_id(self.results, key)
            mv = MauveEvaluator(featurize_model_name = 'Qwen/Qwen3-Embedding-8B')
            summary_V3['mauve_total'] = {}
            summary_V3['mauve_total_detailed'] = {}
            for key in keys:
                summary_V3['mauve_total'][key] = {}
                summary_V3['mauve_total_detailed'][key] = {}
                synth_text = [x[key] for x in self.results]
                
                for split in self.compare_set.keys(): # 'train' and 'test'
                    real_text = [self.compare_set[split][self.class_labels_to_id[cls]] for cls in self.class_labels]
                    real_text = [item for sublist in real_text for item in sublist]
                    
                    res = mv.compute_mauve(synth_text, real_text, mauve_scaling_factor=5, verbose=False, seed=self.seed)
                    summary_V3['mauve_total'][key][split] = res['mauve']
                    summary_V3['mauve_total_detailed'][key][split] = res
            
            save_json(summary_V3, summary_path_V3)
            
    
    def finalize(self, save_path = None):
        raw_results_path = save_path + 'raw_results.json'
        
        if not self.eval_only:
            save_json(self.results, raw_results_path)
        else:
            self.results = load_json(raw_results_path)

        summary = {'quality_mean': None,
                   'mauve': None,
                   'mauve_avg': None,
                   'mauve_total': None,
                   'ICL': None,
                   'BERT': None,
                   'sample': None,
                   'sample_full_prompt': None,
                   'details': None,
                   'mauve_detailed': None,
                   'mauve_total_detailed': None,
                   }
        


        summary_path_V1 = os.path.join(save_path, f'synthetic_data_summary.json')
        summary_path_V2 = os.path.join(save_path, f'synthetic_data_summary_v2.json')
        summary_path_V3 = os.path.join(save_path, f'synthetic_data_summary_v3.json')
        
        return_now = False
        if os.path.exists(summary_path_V2):
            self.convert_V2_to_V3(summary_path_V2, summary_path_V3, summary)
            return_now = True        
        elif os.path.exists(summary_path_V1):
            self.convert_V1_to_V2(summary_path_V1, summary_path_V2)
            self.convert_V2_to_V3(summary_path_V2, summary_path_V3, summary)
            return_now = True

        
        if return_now:
            print('Conversion done. Exiting now.')
            # raise Exception('Converted old summary to new format. No need to re-evaluate.')
            return
        
                 
        

        generated_data = [x['text'] for x in self.results]
        keys = ['text']
        if self.improve_quality:
            
            print('improving quality of generated data...')
            qi_batch_size = 4
            qi = QualityImprover(model_name='Qwen/Qwen3-4B-Instruct-2507', thinking_budget=0)
            improved_results = qi.improve(generated_data, bs=qi_batch_size)
            for i in range(len(self.results)):
                self.results[i]['polished'] = improved_results[i]
            keys = ['text', 'polished']
            
        print('calculating metrics...')
        print('1. Quality ...')

        QE = qe.QualityEvaluator('Qwen/Qwen3-4B-Instruct-2507', thinking_budget=0)
        
        quality_scores = {}
        for key in keys:
            texts = [x[key] for x in self.results]
            quality_scores[key] = QE.evaluate(texts, bs=8)
        
        for i in range(len(self.results)):
            self.results[i][f'quality_score'] = {key: quality_scores[key][i] for key in keys}
        
        del QE
        gc.collect()
        torch.cuda.empty_cache()

        summary['quality_mean'] = {key: np.mean([self.results[i][f'quality_score'][key]['overall'] for i in range(len(self.results))]) for key in keys}
        
        print('2. Mauve ...')
        
        compare_set = self.compare_set
        mv = MauveEvaluator(featurize_model_name = 'Qwen/Qwen3-Embedding-8B')

        summary['mauve'] = {}
        summary['mauve_detailed'] = {}
        summary['mauve_avg'] = {}
        summary['mauve_total'] = {}
        summary['mauve_total_detailed'] = {}
        
        for key in keys:
            summary['mauve'][key] = {}
            summary['mauve_detailed'][key] = {}
            summary['mauve_avg'][key] = {}
            
            for cls in self.class_labels:
                class_id = self.class_labels_to_id[cls]
                synth_text = [x[key] for x in self.results if x['class'] == cls]
                summary['mauve'][key][f'{cls}'] = {}
                summary['mauve_detailed'][key][f'{cls}'] = {}
                
                for split in compare_set.keys(): # 'train' and 'test'
                    real_text = compare_set[split][class_id]
                    res = mv.compute_mauve(synth_text, real_text, mauve_scaling_factor=5, verbose=False, seed=self.seed)
                    summary['mauve'][key][f'{cls}'][split] = res['mauve']
                    summary['mauve_detailed'][key][f'{cls}'][split] = res
            
            
            synth_text = [x[key] for x in self.results]
            summary['mauve_total'][key] = {}
            summary['mauve_total_detailed'][key] = {}
            for split in compare_set.keys(): # 'train' and 'test'
                real_text = [compare_set[split][self.class_labels_to_id[cls]] for cls in self.class_labels]
                real_text = [item for sublist in real_text for item in sublist]
                res = mv.compute_mauve(synth_text, real_text, mauve_scaling_factor=5, verbose=False, seed=self.seed)
                summary['mauve_total'][key][split] = res['mauve']
                summary['mauve_total_detailed'][key][split] = res
            
            
            for split in compare_set.keys(): # 'train' and 'test'            
                summary['mauve_avg'][key] = {split: np.mean([summary['mauve'][key][f'{cls}'][split] for cls in self.class_labels]) for split in compare_set.keys()}
                

        del mv
        gc.collect()
        torch.cuda.empty_cache()

        synthetic_data = {}
        for key in keys:
            synthetic_data[key] = self.categorize_results_texts_by_class_id(self.results, key)
        
        print('3. ICL evaluation...')
        
        icl_evaluator = ICLEvaluator('Qwen/Qwen3-8B', compare_set['train'], compare_set['test'], self.ids_to_class_labels, self.seed)
        
        icl_results = {}
        for key in keys:
            icl_results[key] = icl_evaluator.evaluate(synthetic_data[key])
        
        summary['ICL'] = {key: icl_results[key] for key in keys}
        
        del icl_evaluator
        gc.collect()
        torch.cuda.empty_cache()
        
        print('4. BERT evaluation...')
        bert_evaluator = BERTEvaluator(compare_set['train'], compare_set['test'], epochs=5)
        bert_results = {}
        for key in keys:
            bert_results[key] = {}
            for repeat in range(10):
                bert_results[key][repeat] = bert_evaluator.evaluate(synthetic_data[key], seed=self.seed * 100 + repeat)
        summary['BERT'] = {key: bert_results[key] for key in keys}
        
        del bert_evaluator
        gc.collect()
        torch.cuda.empty_cache()
        
        print('5. Finalizing ...')

        summary['sample'] = {'system': self.sample_s, 'user': self.sample_u, 'assistant': self.sample_a}
        summary['sample_full_prompt'] = self.sample_full_prompt
        
        summary['details'] = self.results
        save_json(summary, summary_path_V3)
        
        print(f'Summary saved to {summary_path_V3}')
        
    def get_unique_name(self):
        return f'Few-shot Baseline'
    
    def get_max_len(self):
        return 512
    
    def get_progress(self):
        return self.counter / self.total_length if self.total_length > 0 else 1.0

    def prepare_inference(self, model, tokenizer):
        return model, tokenizer
    
    def modify_language_model(self, model, tokenizer):
        return model, tokenizer
    
    def clean_cache(self):
        import torch
        torch.cuda.empty_cache()
        return
    
    def get_stopping_criteria(self):
        # return None
        return self.prompt_managers[self.class_labels[self.current_class_index]].get_stopping_criteria()
    
    def get_prompt_template_epsilon_delta(self):
        if self.n_shots > 0:
            epsilon, delta = float('inf'), 1.0
        elif self.n_fixed_shots > 0:
            epsilon, delta = self.fixed_shots_epsilon, self.fixed_shots_delta
        else:
            epsilon, delta = 0.0, 0.0
        return epsilon, delta