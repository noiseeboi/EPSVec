from typing_extensions import override
from methods.Few import Few
from my_utils import load_json
from calculate_eps import VIPrivacyAccountant

class VI(Few):
    def __init__(self, eval_only, dataset, prompt_style, count_per_class, bs, n_fixed_shots, fixed_shots_epsilon, fixed_shots_delta, 
                 vec_target_tokens, vec_per_cluster_count, vec_normalization, vec_type, vec_temp, vec_drop_threshold, injection_reweighting:bool, injection_coeff:float, injection_layers:str, injection_location:int, 
                 model_short_name:str, seed:int, epsilon:float, delta:float, quality_threshold:int, improve_quality:bool, noise_scale:float, noise_layers:str):
        super().__init__(eval_only, dataset, prompt_style, count_per_class, bs, n_shots=0, n_fixed_shots=n_fixed_shots, fixed_shots_epsilon=fixed_shots_epsilon, fixed_shots_delta=fixed_shots_delta, seed=seed, quality_threshold=quality_threshold, improve_quality=improve_quality)
        self.vec_target_tokens = vec_target_tokens
        self.vec_per_cluster_count = vec_per_cluster_count
        self.vec_normalization = vec_normalization
        self.vec_type = vec_type
        self.vec_temp = vec_temp
        self.vec_drop_threshold = vec_drop_threshold
        
        self.injection_reweighting = injection_reweighting
        self.injection_coeff = injection_coeff
        self.injection_layers = injection_layers
        self.injection_location = injection_location
        self.model_short_name = model_short_name
        
        self.epsilon = epsilon
        self.delta = delta
        
        self.layers_list = None
        
        self.noise_scale = noise_scale
        self.noise_layers = noise_layers
        
        assert self.epsilon >= self.fixed_shots_epsilon, "Epsilon for VI must be greater than or equal to that of the fixed-shot prompt template."
        assert self.delta >= self.fixed_shots_delta, "Delta for VI must be greater than or equal to that of the fixed-shot prompt template."
     
    def get_vi_shared_path_name(self):
        nois_str = ''
        if self.noise_scale > 0:
            nois_str = f'ns_{self.noise_scale}_nl_{self.noise_layers}'
        
        return f'eps_{self.epsilon}_delta_{self.delta}/vtt_{self.vec_target_tokens}_vpc_{self.vec_per_cluster_count}_vn_{self.vec_normalization}_vt_{self.vec_type}_vtemp_{self.vec_temp}_vdt_{self.vec_drop_threshold}/itw_{self.injection_reweighting}_ilo_{self.injection_location}/il_{self.injection_layers}_ic_{self.injection_coeff}{nois_str}/feps_{self.fixed_shots_epsilon}_fdelta_{self.fixed_shots_delta}/'

    def get_path_name(self):
        return self.get_shared_path_name() + f'VI_{self.n_fixed_shots}/' + self.get_vi_shared_path_name()
    
    def get_unique_name(self):
        return f'Vector Injection'
    
    @override
    def prepare_inference(self, model, tokenizer):
        from steer_manager import get_steer_fn, load_vectors
        import torch

        
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        
        self.vectors = {}
        for cls in self.class_labels:

            clipping = True if self.epsilon != float('inf') else False
            model_short_name_with_temperature = f"{self.model_short_name}_temperature_{self.vec_temp}"
            base_path_name = f'steer_vectors/{model_short_name_with_temperature}/Seed_{self.seed}/{self.dataset}/{self.prompt_style}/c_{1}_perc_{self.vec_per_cluster_count}_rce_{0.0}_rcd_{0.0}_negc_{self.count_per_class}/sc_Few_fs_{self.n_fixed_shots}_shots_1_eps_{self.fixed_shots_epsilon}_delta_{self.fixed_shots_delta}_dt_{self.vec_drop_threshold}/{self.vec_target_tokens}/vt_{self.vec_type}_norm_{self.vec_normalization}_clip_{clipping}/{cls}/{0}/' 
            
            steer_vecs, steer_layers, steer_locs, steer_tokens = load_vectors(base_path_name, num_hidden_layers, hidden_size, self.injection_layers, self.injection_location)
            
            self.layers_list = steer_layers
            
            if self.epsilon != float('inf'):
                privacy_noise_scale = self.get_noise_scale()
                for i in range(len(self.layers_list)):
                    noise = torch.randn_like(steer_vecs[i]) * privacy_noise_scale[self.layers_list[i]]
                    steer_vecs[i] = steer_vecs[i] + noise
                    
            if self.vec_normalization == 'after':
                steer_vecs = steer_vecs / torch.norm(steer_vecs, dim=-1, keepdim=True)
            
            
            if self.injection_reweighting:
                layer_weights = load_json(f'methods/hyperparams/{self.model_short_name}.json')['scale']
                for i, l in enumerate(steer_layers):
                    steer_vecs[i] = steer_vecs[i] * layer_weights[l]
            
            steer_vecs = steer_vecs * self.injection_coeff
            steer_vecs = steer_vecs.to(model.device)
            
            noise_layers = None
            if self.noise_scale > 0:
                if self.noise_layers == 'all':
                    noise_layers = list(range(num_hidden_layers))
                elif self.noise_layers == 'injection':
                    noise_layers = steer_tokens
                else:
                    noise_layers = list(map(int, self.noise_layers.split(',')))
            
            self.vectors[cls] = get_steer_fn(steer_vecs, steer_layers, steer_locs, steer_tokens, hidden_size, 'add', False, self.noise_scale, noise_layers)
        return model, tokenizer
    
    @override
    def modify_language_model(self, model, tokenizer):
        
        steer_fn = self.vectors[self.class_labels[self.current_class_index]]
        model.set_steer_fn(steer_fn)
        return model, tokenizer
    
    @override
    def clean_cache(self):
        del self.vectors
        return super().clean_cache()
    
    def get_noise_scale(self):
        
        remaining_epsilon = self.epsilon - self.fixed_shots_epsilon
        remaining_delta = self.delta - self.fixed_shots_delta
        
        accountant = VIPrivacyAccountant(remaining_epsilon, remaining_delta, self.dataset, self.vec_per_cluster_count * 1 * 1, # first 1 is number of shots used to create prompt during prompt construction, second 1 is number of clusters per class
                                         self.vec_per_cluster_count, self.model_short_name, self.layers_list, self.vec_normalization)
        
        noise_scales = accountant.get_noise_scales()
        return noise_scales