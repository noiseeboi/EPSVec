from methods.Few import Few
from copy import deepcopy

class Real(Few):
    def __init__(self, eval_only, dataset, prompt_style, count_per_class, bs, split, seed):
        super(Real, self).__init__(eval_only, dataset, 'ptz', count_per_class, bs, n_shots=0, n_fixed_shots=0, fixed_shots_epsilon=0.0, fixed_shots_delta=0.0, seed=seed, quality_threshold=0, improve_quality=False)
        self.prompt_style = prompt_style
        self.split = split
        self.seed = seed
        real_data = deepcopy(self.compare_set[split])
        self.results = [{'text': data, 'class': self.ids_to_class_labels[k], 'raw_text': data} for k in real_data.keys() for data in real_data[k]]

    def get_path_name(self):
        return self.get_shared_path_name() + f'Real_{self.split}/eps_inf_delta_1.0/'
    