import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, BitsAndBytesConfig
import numpy as np
from tqdm.auto import tqdm
import json
import re
from typing import Dict, Any, List

SYSTEM_PROMPT = (
    "You are a precise writing-quality evaluator. "
    "Evaluate the provided text and return ONLY a valid JSON object that follows this schema:\n"
    "{\n"
    '  "fluency": <integer 1-5>,\n'
    '  "grammar": <integer 1-5>,\n'
    '  "coherence": <integer 1-5>,\n'
    # '  "readability": <integer 1-5>,\n'
    '  "overall": <integer 1-10>,\n'
    # '  "rationale": "<=60 words, concise justification>"\n'
    "}\n"
    "Rules: 1) Output pure JSON (no markdown, no extra text). "
    "2) Use integers only for scores."
)

EVAL_INSTRUCTIONS = (
    "Evaluate the following synthetic text using G-Eval-style criteria:\n"
    "- Fluency (1–5): smoothness and flow; natural phrasing, no awkwardness.\n"
    "- Grammar (1–5): correctness of syntax, tense, agreement, punctuation.\n"
    "- Coherence (1–5): logical organization; ideas connect and progress sensibly.\n"
    # "- Readability (1–5): clarity and ease for a general audience; appropriate vocabulary.\n"
    "- Overall (1–10): holistic quality as writing (not factual accuracy).\n"
)

def extract_json(s: str) -> Dict[str, Any]:
    """
    Robustly extract the first JSON object from a string.
    """
    # If the model behaved, the whole output is JSON already:
    try:
        return json.loads(s)
    except Exception:
        pass

    # Strip code fences if any:
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE | re.DOTALL)

    # Try first/last brace heuristic:
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            # As a last resort, try to remove trailing commas
            candidate2 = re.sub(r",\s*([\]}])", r"\1", candidate)
            return json.loads(candidate2)
        except:
            pass
    
    return {'fluency': 0, 'grammar': 0, 'coherence': 0,  'overall': 0} #'readability': 0,, 'rationale': "Failed to parse the input."}



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

class QualityEvaluator():
    def __init__(self, model_name, thinking_budget = 0, quantization = False):
        # load the tokenizer and the model
        assert 'qwen3' in model_name.lower(), 'Only Qwen3 models are supported for quality evaluation.'
        self.is_thinking_model = 'instruct' not in model_name.lower()
        self.thinking_budget = thinking_budget
        assert thinking_budget == 0 or self.is_thinking_model, "The model does not support thinking mode."
        self.quantization = quantization
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if quantization:
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_type=torch.float16
            )
        
            LLM = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="cuda"
            )
        else:
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
            prompt = (
                f"{EVAL_INSTRUCTIONS}\n"
                "Here is the text to evaluate delimited by triple fences:\n"
                "```text\n"
                f"{shot}\n"
                "```\n"
                "Return ONLY the JSON object."
            )
            all_prompts.append(prompt)

        return all_prompts

    def evaluate(self, shots, bs=4, verbose=True):
        thinking_budget = self.thinking_budget
        
        
        prompts = self.make_prompts(shots)

        results = []
        
        num_batches = len(prompts) // bs + (1 if len(prompts) % bs != 0 else 0)
        iterator = tqdm(range(num_batches)) if verbose else range(num_batches)
        for i in iterator:
            batch = prompts[i * bs:(i + 1) * bs]

            chat_batch = []
            for prompt in batch:
                ch = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
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
                outputs = self.LLM.generate(**tokenized, max_new_tokens=50)
                preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
                
                new_preds = []
                for p in preds:
                    if self.is_thinking_model:
                        p = p.split('</think>')[-1]
                        
                    else:
                        p = p.split('<|im_start|>assistant')[-1]
                    
                    new_preds.append(p.split('<|im_end|>')[0].strip())
                
                results.extend(new_preds)
        
        parsed_results = []
        for r in results:
            parsed_results.append(extract_json(r))

        return parsed_results
    
if __name__ == "__main__":
    model_name = "Qwen/Qwen3-8B"
    quality_improver = QualityEvaluator(model_name)
    
    shots = [
        "Don buy ths book. Its full of erors and the story dosnt make sens. Abb A A A aA A. Garbage Iot Used Water!!!",
        "The quik brown fox jumpd over the lazy dogg. It was a sunny dayy.",
        "I enjoyed the movie. It was great!",
        "The morning was there and things were kind of happening. The sky was getting lighter but not in a special way, just sort of changing colors slowly. I walked a bit and thought about stuff but nothing really clear came to mind. There were cars sometimes, and the air felt okay, maybe a little cold. It was quiet, though not completely, because you could still hear small sounds if you listened. Mostly it was just another start to a day, nothing too important going on, and I wasn’t sure what to think about it.",
        "In the quiet hour before dawn, when the world is suspended between shadow and light, a peculiar stillness settles over the streets. It is a moment that feels almost borrowed, as though time has paused to gather its thoughts. The air, cool and unhurried, carries the faintest trace of possibility—an unopened letter waiting on the edge of discovery. Gradually, the horizon softens, and the first strokes of morning unfurl across the sky, revealing colors that seem to exist nowhere else. In this gentle unveiling, even the most ordinary things—the curve of a railing, the hush of distant trees—take on a quiet significance, reminding us that meaning often appears not in grand gestures but in the subtle spaces we rarely stop to notice.",
    ]
    
    improved_texts = quality_improver.evaluate(shots, bs=2, thinking_budget=1000)
    
    for original, improved in zip(shots, improved_texts):
        print(f"Original: {original}")
        print(f"Improved: {improved}")
        print()
