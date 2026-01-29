from typing_extensions import override
from methods.VI import VI
from my_utils import load_json
from calculate_eps import VIPrivacyAccountant

class CVI(VI):
    def __init__(self, eval_only, dataset, prompt_style, count_per_class, bs, 
                 n_fixed_shots, fixed_shots_epsilon, fixed_shots_delta, 
                 vec_target_tokens, vec_per_cluster_count, vec_normalization, vec_type, vec_temp, vec_drop_threshold,
                 injection_reweighting:bool, injection_coeff:float, injection_layers:str, injection_location:int, model_short_name:str, seed:int, 
                 n_clusters:int, clustering_eps:float, clustering_delta:float, epsilon:float, delta:float, 
                 quality_threshold:int, improve_quality:bool,
                 noise_scale:float, noise_layers:str                 
                 ):
        super().__init__(eval_only, dataset, prompt_style, count_per_class, bs, n_fixed_shots, fixed_shots_epsilon, fixed_shots_delta, vec_target_tokens, vec_per_cluster_count, vec_normalization, vec_type, vec_temp, vec_drop_threshold, injection_reweighting, injection_coeff, injection_layers, injection_location, model_short_name, seed, epsilon, delta, quality_threshold, improve_quality, noise_scale, noise_layers)
        self.n_clusters = n_clusters
        
        self.clustering_eps = clustering_eps
        self.clustering_delta = clustering_delta
        
        self.in_cluster_counter = 0
        self.current_cluster_index = 0
        self.total_cluster_count = n_clusters
        self.count_per_cluster = count_per_class // n_clusters
        assert self.count_per_cluster * n_clusters == count_per_class, "count_per_class must be divisible by n_clusters"
        
        assert self.clustering_eps + self.fixed_shots_epsilon <= self.epsilon, f"Sum of clustering epsilon and fixed-shots epsilon must be less than or equal to total epsilon, but got {self.clustering_eps} + {self.fixed_shots_epsilon} > {self.epsilon}"
        if self.delta < 1.0:
            assert self.clustering_delta + self.fixed_shots_delta <= self.delta, f"Sum of clustering delta and fixed-shots delta must be less than or equal to total delta, but got {self.clustering_delta} + {self.fixed_shots_delta} > {self.delta}"
        
    def get_path_name(self):        
        return self.get_shared_path_name() + f'CVI_{self.n_fixed_shots}_{self.n_clusters}/' + self.get_vi_shared_path_name() + f'ceps_{self.clustering_eps}_cdelta_{self.clustering_delta}/'
    
    def get_unique_name(self):
        return f'Clustering Vector Injection'
    
    @override
    def prepare_inference(self, model, tokenizer):
        from steer_manager import get_steer_fn, load_vectors
        import torch

        
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        
        self.vectors = {}
        for cls in self.class_labels:
            self.vectors[cls] = {}
            for cluster_id in range(self.n_clusters):
                clipping = True if self.epsilon != float('inf') else False
                model_short_name_with_temperature = f"{self.model_short_name}_temperature_{self.vec_temp}"
                base_path_name = f'steer_vectors/{model_short_name_with_temperature}/Seed_{self.seed}/{self.dataset}/{self.prompt_style}/c_{self.n_clusters}_perc_{self.vec_per_cluster_count}_rce_{self.clustering_eps}_rcd_{self.clustering_delta}_negc_{self.count_per_class}/sc_Few_fs_{self.n_fixed_shots}_shots_1_eps_{self.fixed_shots_epsilon}_delta_{self.fixed_shots_delta}_dt_{self.vec_drop_threshold}/{self.vec_target_tokens}/vt_{self.vec_type}_norm_{self.vec_normalization}_clip_{clipping}/{cls}/{cluster_id}/'
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
                
                self.vectors[cls][cluster_id] = get_steer_fn(steer_vecs, steer_layers, steer_locs, steer_tokens, hidden_size, 'add', False, self.noise_scale, noise_layers)

        return model, tokenizer
    
    @override
    def modify_language_model(self, model, tokenizer):
        steer_fn = self.vectors[self.class_labels[self.current_class_index]][self.current_cluster_index]
        model.set_steer_fn(steer_fn)
        return model, tokenizer
    
    
    @override
    def get_preferred_batch_size(self):
        preferred_batch_size = self.get_batch_size()
        bs = min(preferred_batch_size, self.count_per_cluster - self.in_cluster_counter)
        return bs
    
    @override
    def update_counters(self, len_responses):        
        super().update_counters(len_responses)

        self.in_cluster_counter += len_responses
        if self.in_cluster_counter >= self.count_per_cluster:
            self.in_cluster_counter = 0
            self.current_cluster_index = (self.current_cluster_index + 1) % self.total_cluster_count
    
    
    def get_noise_scale(self):
        remaining_epsilon = self.epsilon - self.fixed_shots_epsilon - self.clustering_eps
        remaining_delta = self.delta - self.fixed_shots_delta - self.clustering_delta
        
        accountant = VIPrivacyAccountant(remaining_epsilon, remaining_delta, self.dataset, self.vec_per_cluster_count * 1 * self.n_clusters, # first 1 is number of shots used to create prompt during prompt construction
                                         self.vec_per_cluster_count, self.model_short_name, self.layers_list, self.vec_normalization)
        
        noise_scales = accountant.get_noise_scales()
        return noise_scales