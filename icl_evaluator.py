import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
import numpy as np
from tqdm.auto import tqdm

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
                generated_tokens = input_ids[i, self.original_token_lens:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                if not any(stop_string in generated_text for stop_string in self.stop_strings):
                    return False
            return True
            
            

    return [StopOnString(stop_strings, tokenizer, original_token_lens)]

ICL_prompt = '''Consider the following examples with their labels:

{shots}
Now classify the following. Just output the label with no explanation or punctuations.

Text: {testsample}
Label:
'''

shot_template = '''Text: {sample}
Label: {label}
'''

class ICLEvaluator():
    def __init__(self, model_name, real_train_data, real_test_data, id_to_class, seed, thinking_budget=0):
        assert 'qwen3' in model_name.lower(), 'Only Qwen3 models are supported for quality improvement.'
        self.is_thinking_model = 'instruct' not in model_name.lower()
        self.thinking_budget = thinking_budget
        assert thinking_budget == 0 or self.is_thinking_model, "The model does not support thinking mode."
        
        self.id_to_class = id_to_class
        self.class_to_id = {v: k for k, v in id_to_class.items()}
        self.seed = seed

        self.real_train_data = real_train_data
        self.real_test_data = real_test_data

        
        # load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        LLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cuda"
        )   
        LLM.eval() 
        self.tokenizer = tokenizer
        self.LLM = LLM

    def truncate_to_sentence_tokens(self, text: str, max_tokens: int) -> str:
        tokenizer = self.tokenizer
        
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


    def make_prompts(self, train_data, test_data, n_shots=1, seed=0):
        # Set the random seed for reproducibility
        random_number_generator = np.random.default_rng(seed)
        
        all_prompts = []
        all_labels = []
        for test_key in test_data.keys():
            for instance in test_data[test_key]:
                # randomly sample n_shots samples from each train class
                train_shots = []
                
                for key in train_data.keys():
                    sampled_indices = np.random.choice(len(train_data[key]), size=n_shots, replace=True)
                    for idx in sampled_indices:
                        td = train_data[key][idx]
                        td = self.truncate_to_sentence_tokens(td, max_tokens=800)
                        train_shots.append(shot_template.format(sample=td, label=self.id_to_class[key]))
                random_number_generator.shuffle(train_shots)
                
                ts = self.truncate_to_sentence_tokens(instance, max_tokens=800)
                user_prompt = ICL_prompt.format(shots="\n".join(train_shots), testsample=ts)
                all_prompts.append(user_prompt)
                all_labels.append(self.id_to_class[test_key])

        return all_prompts, all_labels

    def evaluate_(self, train_data, test_data, n_shots=1, seed = 0, bs = 4):
        
        thinking_budget = self.thinking_budget
        prompts, labels = self.make_prompts(train_data, test_data, n_shots=n_shots, seed=seed)

        correct = 0
        
        num_batches = len(prompts) // bs + (1 if len(prompts) % bs != 0 else 0)
        for i in tqdm(range(num_batches)):
            batch = prompts[i * bs:(i + 1) * bs]
            batch_labels = labels[i * bs:(i + 1) * bs]

            chat_batch = []
            for prompt in batch:
                ch = [{'role': 'user', 'content': prompt}]
                if self.is_thinking_model:
                    enable_thinking = thinking_budget > 0
                    prompt = self.tokenizer.apply_chat_template(ch, tokenize=False, add_generation_prompt=True, enable_thinking = enable_thinking)
                else:
                    prompt = self.tokenizer.apply_chat_template(ch, tokenize=False, add_generation_prompt=True)
                    
                
                bos_token = self.tokenizer.bos_token
                if bos_token:
                    prompt = prompt.replace(bos_token, '')
                chat_batch.append(prompt)

            with torch.no_grad():
                if thinking_budget > 0:
                    # First, let the model think
                    tokenized = self.tokenizer(chat_batch, return_tensors="pt", padding=True).to("cuda")
                    original_token_lens = tokenized['input_ids'].shape[1]
                    outputs = self.LLM.generate(**tokenized, max_new_tokens=thinking_budget, do_sample=False, stopping_criteria=get_stopping_criteria(['</think>'], self.tokenizer, original_token_lens))
                    thought_preds = self.tokenizer.batch_decode(outputs[:, original_token_lens:], skip_special_tokens=False)
                    chat_batch_thoughts = []
                    for original_prompt, thought_pred in zip(chat_batch, thought_preds):
                        thought = thought_pred.split('<|im_end|>')[0].strip()
                        if '</think>' not in thought:
                            thought += '\n</think>'
                        chat_batch_thoughts.append(original_prompt + thought)
                    chat_batch = chat_batch_thoughts
                
                # Now generate the final improved text
                
                tokenized = self.tokenizer(chat_batch, return_tensors="pt", padding=True).to("cuda")
                outputs = self.LLM.generate(**tokenized, max_new_tokens=10)
                preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
                
                new_preds = []
                for p in preds:
                    if self.is_thinking_model:
                        p = p.split('</think>')[-1]
                        
                    else:
                        p = p.split('<|im_start|>assistant')[-1]
                    
                    new_preds.append(p.split('<|im_end|>')[0].strip())
                
                preds = new_preds
                
                for p, l in zip(preds, batch_labels):
                    # print('pred', p, 'label', l)
                    if l in p:
                        correct += 1

        accuracy = correct / len(prompts) if prompts else 0
        return accuracy
    
    def evaluate(self, synthetic_data):
        TRTS = self.evaluate_(self.real_train_data, synthetic_data, seed=self.seed)
        TSTR = self.evaluate_(synthetic_data, self.real_test_data, seed=self.seed)

        return {'TRTS': TRTS, 'TSTR': TSTR}


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-4B"
    real_train_data = {'0': ['The cat sat on the mat.', 'A dog barked loudly.'], '1': ['The sun is shining.', 'It is raining today.']}
    real_test_data = {'0': ['A feline is resting on a rug.'], '1': ['The weather is sunny.']}
    id_to_class = {'0': 'animal activity', '1': 'weather description'}
    icl_evaluator = ICLEvaluator(model_name, real_train_data=real_train_data, real_test_data=real_test_data, id_to_class=id_to_class, seed=42)
    synthetic_data = {'0': ['A kitten is lying on a carpet.'], '1': ['Today is a bright and sunny day.']}
    results = icl_evaluator.evaluate(synthetic_data=synthetic_data, thinking_budget=1000)
    print(results)