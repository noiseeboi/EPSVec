import json
from typing import List, Dict
from copy import deepcopy
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

descriptions = {'imdb': {'singular': 'movie review with a {class_label} sentiment', 'plural': 'movie reviews with {class_label} sentiments'},
                'yelp': {'singular': 'review with a {class_label} sentiment from one domain (e.g., product, food, and service reviews)', 'plural': 'reviews with {class_label} sentiments across different domains (e.g., product, food, and service reviews)'},
                'agnews': {'singular': 'short news article about {class_label}. The sample is in neutral journalistic style, similar to brief newswire summaries.', 'plural': 'short news articles about {class_label}. The samples are in neutral journalistic style, similar to brief newswire summaries.'},
                'dbpedia': {'singular': 'article', 'plural': 'articles'},
                'biorxiv': {'singular': 'abstract section of a journal article on {class_label}. The abstract is a single coherent paragraph starting with a review of the background and objectives, followed by methods, results, and conclusions', 
                            'plural': 'abstract sections of journal articles on {class_label}. Each abstract is a single coherent paragraph starting with a review of the background and objectives, followed by methods, results, and conclusions'},
                'openreview': {'singular': 'review of an ICLR paper with {class_label} recommendation for acceptance', 'plural': 'reviews of ICLR papers with {class_label} recommendation for acceptance'}
}


class PromptGenerator:
    
    def __init__(self, dataset_name, class_label, style, n_shots, n_fixed_shots, fixed_shots_epsilon, fixed_shots_delta):
        assert style in ['gens', 'gen', 'ptz', 'it', 'pt'], f"Invalid style: {style}"
        assert n_shots >= 0 and n_fixed_shots >= 0, "n_shots must be non-negative"

        plurality = 'plural' if style in ['ptz'] else 'singular'
        self.domain_description = descriptions[dataset_name][plurality].format(class_label=class_label)
        self.class_label = class_label
        self.style = style
        self.n_shots = n_shots
        self.n_fixed_shots = n_fixed_shots
        
        if self.style in ['it', 'pt']:
            assert n_shots + n_fixed_shots > 0, f"For 'it' and 'pt' styles, got n_shots={n_shots}, n_fixed_shots={n_fixed_shots}"

        
        if self.n_fixed_shots > 0:
            path = f'fixed_shots/{dataset_name}/eps_{fixed_shots_epsilon}_delta_{fixed_shots_delta}/{dataset_name}_fixed_few_shot_samples.json'
            self.fixed_shot_samples: Dict[str, List[str]] = load_json(path)[class_label]
            assert len(self.fixed_shot_samples) >= self.n_fixed_shots, f"Not enough fixed shots for class {class_label}, dataset {dataset_name}, in {path}"

        self.prompt_header = self.get_prompt_header()
        self.prompt_footer = self.get_prompt_footer()    
                    
    
    def get_prompt_footer(self):
        if self.style == 'it':
            return f'\n\nPlease give another one. No formatting or explanations.'
        elif self.style in ['gen', 'gens']:
            return '\nJust provide the sample without any additional text or formatting.'
        elif self.style in ['pt', 'ptz']:
            return ''
    
    def prepare_fewshot_samples(self, bs, few_shot_samples):
        if self.n_shots + self.n_fixed_shots < 1:
            return [''] * bs
        few_shot_samples = deepcopy(few_shot_samples)
        assert len(few_shot_samples) == bs * self.n_shots, "Not enough few shot samples provided"

        few_shots_samples_with_template = []
        few_shot_template = self.get_few_shot_template()
        for b in range(bs):
            fixed_shots = []
            if self.n_fixed_shots > 0:
                fixed_shots = self.fixed_shot_samples[:self.n_fixed_shots]
            
            shots = fixed_shots + [few_shot_samples[b * self.n_shots + i] for i in range(self.n_shots)]

            fts = '\n\n'.join([few_shot_template.format(sample=shot) for shot in shots])
            few_shots_samples_with_template.append(fts)

        return few_shots_samples_with_template

    def get_user_prompt(self, bs, few_shot_samples):
        header = self.get_prompt_header()
        footer = self.get_prompt_footer()
        few_shot_samples = self.prepare_fewshot_samples(bs, few_shot_samples)
        return [(header + few_shot_samples[i] + footer) for i in range(bs)]
    
    def get_few_shot_template(self):
        if self.style == 'it':
            return 'Text: {sample}'
        elif self.style == 'ptz':
            p = '<begin>'
            p += f'\nLabel: {self.class_label}\n'
            p += 'Text: {sample}\n</end>'
            return p
        elif self.style == 'pt':
            p = '```'
            p += f'\n{self.class_label}\n'
            p += '{sample}\n```'
            return p
        elif self.style in ['gen', 'gens']:
            return 'Sample: {sample}'


    def get_prompt_header(self):
        if self.style == 'it':
            p = f'Here are texts with Label: {self.class_label}.'
            if self.n_shots + self.n_fixed_shots > 0:
                p += '\n\n'
            return p
        elif self.style == 'pt':
            return ''
        elif self.style == 'ptz':
            p = (
                f"Below are several diverse examples of {self.domain_description}.\n"
                "Each example is human-written and enclosed between <begin> and </end> tags.\n"
                "Within each example, the content is structured into two fields:\n"
                '- \"Label:\" — describes the category or type of the example\n'
                '- \"Text:\" — contains the corresponding text content\n\n'
                "Here are the examples:"
            )
            if self.n_shots + self.n_fixed_shots > 0:
                p += '\n\n'
            return p
        
        elif self.style in ['gen', 'gens']:
            p = f'Generate a human-like example of {self.domain_description}.'
            if self.n_shots + self.n_fixed_shots > 0:
                p += ' Here are some examples:\n\n'
            return p
    
    def get_system_prompt(self, bs):
        if self.style == 'gens':
            return ['You are a helpful assistant.'] * bs
        elif self.style == 'gen':
            return [''] * bs
        elif self.style == 'ptz':
            return [''] * bs
        elif self.style == 'it':
            return [''] * bs
        elif self.style == 'pt':
            return [''] * bs
    
    
    def get_assistant_prompt(self, bs):
        if self.style == 'it':
            return [''] * bs
        elif self.style == 'pt':
            return [f'```\n{self.class_label}\n'] * bs
        elif self.style == 'ptz':
            return [f'<begin>\nLabel: {self.class_label}\nText:'] * bs
        elif self.style in ['gen', 'gens']:
            return [''] * bs        
        
    def post_process_response(self, response):
        response = response.strip()
        
        if self.style == 'pt':
            if '```' in response:
                response = response.split('```')[0].strip().lstrip()
        elif self.style == 'ptz':
            strs = ['<end>', '</end>', '<begin>', '</begin>', 'Label:', 'Text:']
            for s in strs:
                if s in response:
                    response = response.split(s)[0].strip().lstrip()
        elif self.style == 'it':
            if response.startswith('Text:'):
                response = response.split('Text:')[1].strip().lstrip()
        elif self.style in ['gen', 'gens']:
            if response.startswith('Sample:'):
                response = response.split('Sample:')[1].strip().lstrip()
        
        return response

    def get_stopping_criteria(self):
        if self.style == 'pt':
            return ['```']
        elif self.style == 'ptz':
            return ['</end>', '<end>', '<begin>', '</begin>', 'Label:', 'Text:']
        else:
            return None