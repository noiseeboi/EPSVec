import torch
from steer_vec_utils import extraction_locations, name_to_loc_and_layer
import os

def load_vectors(base_path, num_hidden_layers, hidden_size, steer_layers, steer_locs):
    
    steer_layers = list(map(int, steer_layers.split(','))) if steer_layers != 'all' else list(range(num_hidden_layers))
    steer_locs = list(map(int, steer_locs.split(',')))
    
    steer_vecs = torch.zeros((len(steer_layers), len(steer_locs), hidden_size))
    

    assert os.path.exists(base_path), f"Steer vectors path {base_path} does not exist. Please generate the steer vectors first."
    
    
    for i, layer in enumerate(steer_layers):
        for j, loc in enumerate(steer_locs):
            steer_vecs[i, j] = torch.load(f'{base_path}/layer_{layer}/loc_{loc}/steer_vector.pt')

    return steer_vecs, steer_layers, steer_locs, torch.tensor([-1]) # for now, we only inject to the last token


def get_steer_fn(steer_vector, steer_layers, steer_locs, tokens_slice, hs_size, mode, renormalize, noise_scale=0, noise_layers=None):
    
    assert [steer_loc in extraction_locations.keys() for steer_loc in steer_locs], "Invalid steer locations provided"
    assert (10 not in steer_locs), "Cannot extract attention weights from this function"
    assert noise_scale == 0 or (noise_layers is not None), "If noise_scale > 0, noise_layers must be provided"
    
    # tokens_slice = slice(None)
    if len(steer_vector.shape) == 3:
        steer_vector = steer_vector.unsqueeze(2)
    
    assert steer_vector.shape == (len(steer_layers), len(steer_locs), 1, hs_size)
    
    steer_vector = steer_vector.unsqueeze(0)  # Add batch dimension
    
    names_to_intervene = [extraction_locations[loc].replace("[LID]", str(layer)) for layer in steer_layers for loc in steer_locs]
    names_to_inject_noise = [extraction_locations[loc].replace("[LID]", str(layer)) for layer in noise_layers for loc in steer_locs] if noise_scale > 0 else []
    
    def steer_fn(input_vector, hook):
        name = hook.name
        
        seq_len = input_vector.shape[1]
        if name in names_to_intervene:
            # print(f"Steering {name} with shape {input_vector.shape} and seq_len {seq_len}")
            
            loc, layer = name_to_loc_and_layer(name)
            layer_idx = steer_layers.index(layer)
            loc_idx = steer_locs.index(loc)
            
            uA = input_vector[:, tokens_slice, :] # batch, token, hidden
            
            if renormalize:
                uA_norm = torch.norm(uA, dim=-1, keepdim=True)
            
            v = steer_vector[:, layer_idx, loc_idx, tokens_slice]#.to(input_vector.device)
            # print("v device:", v.device, "uA device:", uA.device)
            if mode == 'replace':
                uA = v
            elif mode == 'add':
                uA = uA + v
            else:
                raise ValueError("Invalid mode. Choose 'replace' or 'add'.")
            
            if renormalize:
                uA = uA / uA.norm(dim=-1, keepdim=True) * uA_norm
            
            input_vector[:, tokens_slice, :] = uA.to(input_vector.dtype)
        
        if noise_scale > 0 and name in names_to_inject_noise:
            # print(f"Adding noise to {name} with shape {input_vector.shape}")
            noise = torch.randn_like(input_vector[:, tokens_slice, :]) * noise_scale
            input_vector[:, tokens_slice, :] = input_vector[:, tokens_slice, :] + noise.to(input_vector.dtype)
            
        return input_vector
    
    return steer_fn

def get_default_steer_fn():
    return lambda input_vector, hook: None