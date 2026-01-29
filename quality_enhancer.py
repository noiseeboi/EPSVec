import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
import numpy as np
from tqdm.auto import tqdm



ICL_prompt = """
You are given a text below that may contain grammatical errors, unclear phrasing, or incoherent structure. Rewrite, prune and polish the text to make it clear, polished, and well-written. 
Provide only the improved version—no explanations.

Text to improve:
{shot}"""

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

class QualityImprover():
    def __init__(self, model_name, thinking_budget = 0):        
        # load the tokenizer and the model
        assert 'qwen3' in model_name.lower(), 'Only Qwen3 models are supported for quality improvement.'
        self.is_thinking_model = 'instruct' not in model_name.lower()
        self.thinking_budget = thinking_budget
        assert thinking_budget == 0 or self.is_thinking_model, "The model does not support thinking mode."
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        LLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cuda"
        )   
        LLM.eval() 
        self.tokenizer = tokenizer
        self.LLM = LLM
        
        
    def make_prompts(self, shots):
        all_prompts = []
        
        for shot in zip(shots):
            prompt = ICL_prompt.format(shot=shot)
            all_prompts.append(prompt)

        return all_prompts

    def improve(self, shots, bs=4):
        thinking_budget = self.thinking_budget

        prompts = self.make_prompts(shots)

        results = []
        
        num_batches = len(prompts) // bs + (1 if len(prompts) % bs != 0 else 0)
        for i in tqdm(range(num_batches)):
            batch = prompts[i * bs:(i + 1) * bs]

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
                outputs = self.LLM.generate(**tokenized, max_new_tokens=len(tokenized['input_ids'][0]))
                preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
                
                new_preds = []
                for p in preds:
                    if self.is_thinking_model:
                        p = p.split('</think>')[-1]
                        
                    else:
                        p = p.split('<|im_start|>assistant')[-1]
                    
                    new_preds.append(p.split('<|im_end|>')[0].strip())
                
                results.extend(new_preds)

        return results
    
if __name__ == "__main__":
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    quality_improver = QualityImprover(model_name)
    
    shots = [
        "Don buy ths book. Its full of erors and the story dosnt make sens. Abb A A A aA A. Garbage Iot Used Water!!!",
        "The quik brown fox jumpd over the lazy dogg. It was a sunny dayy."
    ]
    
    
    improved_texts = quality_improver.improve(shots, bs=2)
    
    for original, improved in zip(shots, improved_texts):
        print(f"Original: {original}")
        print(f"Improved: {improved}")
        print()
