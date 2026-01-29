import numpy as np
from functools import partial
from tqdm.auto import tqdm
from math import exp, sqrt
from scipy.special import erf
from my_utils import load_json

from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import GaussianAccountant  # or RDPAccountant
from math import exp, sqrt
from scipy.special import erf

ERROR=1/2**15

def binary_search(func, low, high, target, error=ERROR, max_iter=20):
    initial_low = low
    initial_high = high
    for _ in range(max_iter):
        mid = (low + high) / 2
        result = func(mid)
        if abs(result - target) < error:
            return mid
        if result < target:
            low = mid
        else:
            high = mid
    raise ValueError(f"Binary search for epsilon could not converge to target {target} within {max_iter} iterations, range [{initial_low}, {initial_high}]. Last low: {low}, high: {high}, mid: {mid}, result: {result}")
    # return (low + high) / 2

def doubling_search(func, target, error=ERROR):
    candidate = 1.0
    while True:
        result = func(candidate)
        if result >= target - error:
            break

        candidate *= 2.0
        
    # Now do binary search between candidate / 2 and candidate
    try:
        return binary_search(func, 0, candidate, target, error, max_iter=int(np.log2(candidate / error) + 5))
    except ValueError:
        raise ValueError(f"Doubling search failed to find an upper bound for target {target}.")

def subsampling_func(initial_epsilon, q):
    """Compute final epsilon after subsampling with rate q."""

    res = np.log(1 + q * (np.exp(initial_epsilon) - 1))
    if res == float('inf') or res == np.nan:
        return np.log(q) + initial_epsilon
    return res

def compute_subsampling(final_epsilon, final_delta, q):
    epsilon = doubling_search(partial(subsampling_func, q=q), final_epsilon, error=ERROR)
    delta = final_delta / q
    return epsilon, delta




def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol = 1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)

    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        
    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma


def exp(power):
    # exponential with overflow guard
    res = np.exp(power)
    if res == float('inf') or res == np.nan:
        raise OverflowError("Exponential overflow")
    return res

class VIPrivacyAccountant:
    def __init__(self, final_epsilon: float, final_delta: float, dataset_name: str, total_data_used_per_class: int, data_used_per_vector_construction_per_class: str, model_name: str, layers_list: list[str], normalization: str):
        self.epsilon = final_epsilon
        self.delta = final_delta
        
        assert normalization in ['before', 'after'], "normalization must be 'before' or 'after'"
        
        per_class_original_dataset_size = load_json(f'methods/hyperparams/{dataset_name}_data_sizes.json')['train_sizes']["0"]
        self.q = total_data_used_per_class / per_class_original_dataset_size
        
        
        m = load_json(f'methods/hyperparams/{model_name}.json')['clip']
        
        self.k = len(layers_list)    
        if normalization == 'after':
            self.Gs = {l: 1 * m[l] / data_used_per_vector_construction_per_class for l in layers_list}
        else:
            self.Gs = {l: 1 / data_used_per_vector_construction_per_class for l in layers_list}

    def get_noise_scales(self):
        epsilon_before_subsampling, delta_before_subsampling = compute_subsampling(self.epsilon, self.delta, self.q)
        print(f"VI Privacy Accountant: epsilon before subsampling: {epsilon_before_subsampling}, delta before subsampling: {delta_before_subsampling}, q: {self.q}, k: {self.k}")
        sigma = get_noise_multiplier(
            target_epsilon=epsilon_before_subsampling,
            target_delta=delta_before_subsampling,
            sample_rate=1.0,
            steps=self.k,
            accountant="prv",                  # or "rdp"
        )

        noise_scales = {}
        for k, G in self.Gs.items():
            noise_scales[k] = sigma * G

        return noise_scales

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    final_epsilons = np.linspace(0.0001, 20.0, 160)
    qs = [500/250000, 500/2000]
    final_delta = 1e-5

    
    results = {}
    for q in qs:
        results[q] = []
        for final_epsilon in tqdm(final_epsilons):
            epsilon, delta = compute_subsampling(final_epsilon, final_delta, q)
            results[q].append((epsilon, delta))


    for q in results:
        results[q] = np.array(results[q])
        plt.plot(final_epsilons, results[q][:, 0], label=f'q={q:.5f}')
        
    plt.legend()
    plt.xlabel('final epsilon after subsampling')
    plt.ylabel('value before subsampling')
    plt.grid()

    plt.savefig('subsampling_effects.png')
    plt.show()
    
    plt.clf()
    

    layers = list(range(32))
    dataset_name = 'yelp'
    final_epsilon = 0.1
    final_delta = 1e-5
    data_used_per_vector_construction = 500
    total_data_used = data_used_per_vector_construction * 10
    model_name = 'Llama3.1_8B_PT' #'Llama3.1_8B_PT'
    normalization = 'after'

    pa = VIPrivacyAccountant(final_epsilon, final_delta, dataset_name, total_data_used, data_used_per_vector_construction, model_name, layers, normalization)
    print(pa.get_noise_scales())