import logging
import os
import subprocess
from datetime import datetime
from functools import partial
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from sklearn.linear_model import ElasticNet, LogisticRegression, LinearRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from transformer_lens.hook_points import HookPoint
import random


# Define the dataset class for handling text data
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text


# Log class to handle logging activities
class Log:
    def __init__(self, log_name='probe'):
        filename = f'{log_name}_date-{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}.txt'
        os.makedirs('logs', exist_ok=True)
        self.log_path = os.path.join('logs/', filename)
        self.logger = self._setup_logging()

    def _setup_logging(self):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler(self.log_path),
                                logging.StreamHandler()
                            ])
        return logging.getLogger()


def log_system_info(logger):
    """
    Logs system memory and GPU details.
    """

    def run_command(command):
        """
        Runs a shell command and returns its output.

        Args:
        - command (list): Command and arguments to execute.

        Returns:
        - str: Output of the command.
        """
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return result.stderr

    gpu_info = run_command(['nvidia-smi'])

    if os.name == 'nt':  # windows system
        pass
    else:
        memory_info = run_command(['free', '-h'])
        logger.info("Memory Info:\n" + memory_info)

    logger.info("GPU Info:\n" + gpu_info)


def hf_login(logger):
    load_dotenv()
    try:
        # Retrieve the token from an environment variable
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token is None:
            logger.error("Hugging Face token not set in environment variables.")
            return

        # Attempt to log in with the Hugging Face token
        login(token=token)
        logger.info("Logged in successfully to Hugging Face Hub.")
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")


def find_token_length_distribution(data, tokenizer):
    token_lengths = []
    for text in data:
        tokens = tokenizer.tokenize(text)
        token_lengths.append(len(tokens))

    token_lengths = np.array(token_lengths)
    quartiles = np.percentile(token_lengths, [25, 50, 75])
    min_length = np.min(token_lengths)
    max_length = np.max(token_lengths)

    return {
        "min_length": min_length,
        "25th_percentile": quartiles[0],
        "median": quartiles[1],
        "75th_percentile": quartiles[2],
        "max_length": max_length
    }

def emotion_to_token_ids(emotion_labels, tokenizer):
    some_random_text = "Hello, I am a random text."
    new_batch = [f"{some_random_text} {label}" for label in emotion_labels]

    inputs = tokenizer(
        new_batch,
        padding='longest',
        truncation=False,
        return_tensors="pt",
    )
    label_ids = inputs['input_ids'][:, -1]
    return label_ids

def get_emotion_logits(dataloader, tokenizer, model, ids_to_pick = None, apply_argmax = False):

    probs = []

    for i, (batch_texts, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = tokenizer(
            batch_texts,
            padding='longest',
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        logits = outputs.logits.cpu()

        logits = logits[:, -1, :]
        if not (ids_to_pick is None):
            logits = logits[:, ids_to_pick]

        if apply_argmax:
            logits = torch.argmax(logits, dim=-1)

        probs.append(logits)

    probs = torch.cat(probs, dim=0)
    return probs


def probe(all_hidden_states, labels, appraisals, logger):
    if isinstance(all_hidden_states, torch.Tensor):
        all_hidden_states = all_hidden_states.cpu().numpy()

    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)

    # Normalize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Y_emotion = labels[:, 0]
    Y_appraisals = labels[:, 1:]

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # Probing for emotion (classification)
    try:
        # logger.info(f"Feature matrix shape: {X.shape}")
        # logger.info(f"Target vector shape: {Y_emotion.shape}")

        cv_accuracies = cross_val_score(LogisticRegression(max_iter=2000), X, Y_emotion, cv=kfold, scoring='accuracy')
        classifier = LogisticRegression(max_iter=2000)
        classifier.fit(X, Y_emotion)  # Train on the entire dataset for full model training after CV
        training_accuracy = classifier.score(X, Y_emotion)

        logger.info(f"5-Fold CV Accuracy for emotion category: {cv_accuracies.mean():.4f} ± {cv_accuracies.std():.4f}")
        logger.info(f"Training Accuracy for emotion category: {training_accuracy:.4f}")

        results['emotion'] = {
            'cv_accuracy': cv_accuracies.mean(),
            'cv_std': cv_accuracies.std(),
            'training_accuracy': training_accuracy
        }
    except Exception as e:
        logger.error(f"Error while probing emotion category: {e}")

    # Probing for each appraisal (regression)
    for i, appraisal_name in enumerate(appraisals):
        try:
            Y = Y_appraisals[:, i]
            logger.info(f"Probing appraisal: {appraisal_name}")
            # logger.info(f"Feature matrix shape: {X.shape}")
            # logger.info(f"Target vector shape: {Y.shape}")
            # logger.info(f"Feature 1st 5: {X[:5]}")
            # logger.info(f"Target 1st 5: {Y[:5]}")

            # Define parameter grid for ElasticNet
            param_grid = {
                'alpha': [0.1], #, 1.0, 10.0
                'l1_ratio': [0.1] #, 0.5, 0.9
            }
            
            enet = ElasticNet(max_iter=5000)
            grid_search = GridSearchCV(enet, param_grid, cv=kfold, scoring='r2', n_jobs=-1)
            grid_search.fit(X, Y)
            # enet.fit(X, Y)
            # best_model  = enet
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"Best hyperparameters for '{appraisal_name}': {best_params}")

            cv_mse = cross_val_score(best_model, X, Y, cv=kfold, scoring='neg_mean_squared_error')
            cv_r2 = cross_val_score(best_model, X, Y, cv=kfold, scoring='r2')
            
            training_predictions = best_model.predict(X)
            training_mse = mean_squared_error(Y, training_predictions)
            training_r2 = r2_score(Y, training_predictions)


            logger.info(f"5-Fold CV MSE for '{appraisal_name}': {-cv_mse.mean():.4f} ± {cv_mse.std():.4f}")
            logger.info(f"Training MSE for '{appraisal_name}': {training_mse:.4f}")
            logger.info(f"5-Fold CV R-squared for '{appraisal_name}': {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
            logger.info(f"Training R-squared for '{appraisal_name}': {training_r2:.4f}")
            logger.info("- -"*25)
        except Exception as e:
            logger.error(f"Error while probing appraisal '{appraisal_name}': {e}")
        
        results[appraisal_name] = {
            'training_mse': training_mse,
            'cv_mse': -cv_mse.mean(),
            'cv_mse_std': cv_mse.std(),
            'training_r2': training_r2,
            'cv_r2': cv_r2.mean(),
            'cv_r2_std': cv_r2.std()
        }

    return results


extraction_locations = {1: "model.layers.[LID].hook_initial_hs",
                        2: "model.layers.[LID].hook_after_attn_normalization",
                        3: "model.layers.[LID].hook_after_attn",
                        4: "model.layers.[LID].hook_after_attn_hs",
                        5: "model.layers.[LID].hook_after_mlp_normalization",
                        6: "model.layers.[LID].hook_after_mlp",
                        7: "model.layers.[LID].hook_after_mlp_hs",
                        8: "model.layers.[LID].self_attn.hook_attn_heads",
                        9: "model.final_hook",
                        10: "model.layers.[LID].self_attn.hook_attn_weights",
                        }

def name_to_loc_and_layer(name):
    layer = int(name.split("model.layers.")[1].split(".")[0])
    loc_suffixes = {v.split('.')[-1]:k for k,v in extraction_locations.items()}
    loc = loc_suffixes[name.split(".")[-1]]
    
    return loc, layer

def extract_from_cache(cache_dict_, extraction_layers=[0, 1],
                          extraction_locs=[1, 7],
                          extraction_tokens=[-1]):
    return_value = []

    for layer in extraction_layers:
        return_value.append([])
        for el_ in extraction_locs:
            el = extraction_locations[el_].replace("[LID]", str(layer))
            if el_ != 10: # attention weights should be treated differently
                return_value[-1].append(
                    cache_dict_[el][:, extraction_tokens].cpu())
            else:
                return_value[-1].append(
                        cache_dict_[el][:, :, extraction_tokens].cpu())

        return_value[-1] = torch.stack(return_value[-1], dim=1)
    return_value = torch.stack(return_value, dim=1)
    return return_value
        

def extract_hidden_states(dataloader, tokenizer, model, assistant_tag,
                          extraction_layers=[0, 1],
                          extraction_locs=[1, 7],
                          extraction_tokens=[-1],
                          avg_token_dim = False,
                          do_final_cat = True):
    
    assert [extraction_loc in extraction_locations.keys() for extraction_loc in extraction_locs]    
    assert (10 not in extraction_locs) or len(extraction_locs) == 1
    
    assert extraction_tokens == 'all' or extraction_tokens == 'assistant' or isinstance(extraction_tokens, list)
    
    assert (isinstance(extraction_tokens, list) or (avg_token_dim or not do_final_cat)), "Cannot concatenate the results if all tokens are extracted"
    
    output_attentions = 10 in extraction_locs

    return_values = []
    
    token_slice = extraction_tokens if isinstance(extraction_tokens, list) else slice(None)
    
    for i, batch_texts in tqdm(enumerate(dataloader), total=len(dataloader)):

        inputs = tokenizer(
            batch_texts,
            padding='longest',
            truncation=False,
            return_tensors="pt",
        ).to(model.device)
        
        if extraction_tokens == 'assistant':
            before_tag_lengths = []
            for k, text in enumerate(batch_texts):
                assert assistant_tag in text, f"Assistant tag '{assistant_tag}' not found in text: {text}"
                split_index = text.index(assistant_tag)
                before_tag_text = text[:split_index + len(assistant_tag)]
                after_tag_text = text[split_index + len(assistant_tag):]
                before_tag_tokenized = tokenizer(
                    [before_tag_text],
                    padding='longest',
                    truncation=False,
                    return_tensors="pt",
                    add_special_tokens=True,
                )['input_ids'][0]
                
                after_tag_tokenized = tokenizer(
                    [after_tag_text],
                    padding='longest',
                    truncation=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )['input_ids'][0]
                
                
                assert len(before_tag_tokenized) + len(after_tag_tokenized) == sum(inputs['attention_mask'][k] > 0), \
                    f"Mismatch in token lengths: {len(before_tag_tokenized) + len(after_tag_tokenized)} != {len(inputs['input_ids'][k])} for text: {text}"
                    
                before_tag_lengths.append(len(before_tag_tokenized))
            
            

        with torch.no_grad():
            outputs = model.run_with_cache(**inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions)

        cache_dict_ = outputs[1]
        
        
        
        r = extract_from_cache(cache_dict_, extraction_layers=extraction_layers,
                          extraction_locs=extraction_locs,
                          extraction_tokens=extraction_tokens if isinstance(extraction_tokens, list) else slice(None))
        r = r.cpu()
        
        
        for j in range(len(r)):
            mask = inputs['attention_mask'][j].cpu()
            mask = mask[token_slice]
            v = r[j, :, :, mask == 1]
            
            if extraction_tokens == 'assistant':
                v = v[:, :, before_tag_lengths[j]:]
            
            if avg_token_dim:
                v = v.mean(dim=2, keepdim=True)
            
            return_values.append(v)
        
    if do_final_cat:
        return_values = torch.stack(return_values, dim=0)

    return return_values


def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)



def probe_classification(all_hidden_states, labels, normalize_X = False, fit_intercept = False, handle_imbalanced = False):    
    
    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)
    Y = labels
    
    num_classes = len(set(Y.flatten().tolist()))

    X_train_imbalance, X_test_imbalance, Y_train_imbalance, Y_test_imbalance = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train_imbalance.shape[0]}, Test size: {X_test_imbalance.shape[0]}")
    if normalize_X:
        scaler = StandardScaler(with_std=False)
        X_train_imbalance = scaler.fit_transform(X_train_imbalance)
        X_test_imbalance = scaler.transform(X_test_imbalance)
        
    if handle_imbalanced:
        smote = SMOTE(random_state=42)
        X_train, Y_train = smote.fit_resample(X_train_imbalance, Y_train_imbalance)
        smote = SMOTE(random_state=42)
        X_test, Y_test = smote.fit_resample(X_test_imbalance, Y_test_imbalance)
    else:
        X_train, Y_train = X_train_imbalance, Y_train_imbalance
        X_test, Y_test = X_test_imbalance, Y_test_imbalance

    best_logistic_regression = None
    best_loss = float('inf')
    best_C = None
    
    for c in [-3, -2, -1, 0, 1, 2]:
        C = 10**c
        
        net = LogisticRegression(C = C, fit_intercept=fit_intercept)
        net.fit(X_train, Y_train)
        
        y_probs_test = net.predict_log_proba(X_test)
        loss_test = -y_probs_test[np.arange(y_probs_test.shape[0]), Y_test].mean()
        
        if loss_test < best_loss:
            best_loss = loss_test
            best_logistic_regression = net
            best_C = C
    
    print(f"Best C: {best_C} ------------------")
    net = best_logistic_regression
    y_pred_train = net.predict(X_train)
    y_pred_test = net.predict(X_test)
    
    y_pred_train_imbalance = net.predict(X_train_imbalance)
    y_pred_test_imbalance = net.predict(X_test_imbalance)
      
    if isinstance(Y_train, np.ndarray):
        Y_train = torch.tensor(Y_train)
        Y_test = torch.tensor(Y_test)
        Y_train_imbalance = torch.tensor(Y_train_imbalance)
        Y_test_imbalance = torch.tensor(Y_test_imbalance)
        
    if isinstance(y_pred_train, np.ndarray):
        y_pred_train = torch.tensor(y_pred_train)
        y_pred_test = torch.tensor(y_pred_test)
        y_pred_train_imbalance = torch.tensor(y_pred_train_imbalance)
        y_pred_test_imbalance = torch.tensor(y_pred_test_imbalance)
        
    accuracy_train = (Y_train == y_pred_train).float().mean()
    accuracy_test = (Y_test == y_pred_test).float().mean()
    
    res = {'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test}
    res['weights'] = net.coef_
    res['bias'] = net.intercept_
    
    # add confusion matrix for imbalanced data
    res['confusion_matrix_train'] = confusion_matrix(Y_train_imbalance, y_pred_train_imbalance)
    res['confusion_matrix_test'] = confusion_matrix(Y_test_imbalance, y_pred_test_imbalance)
    
    res['best_C'] = best_C
    res['best_loss'] = best_loss
    
    return res



def probe_regression(all_hidden_states, labels):
    
    if len(labels.shape) == 1:
        labels = labels[:, None]
    
    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)
    Y = labels

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    best_logistic_regression = None
    best_loss = float('inf')
    best_alpha = None
    
    for alpha in [5.0]:
        
        net = Ridge(alpha=5.0)  # ElasticNet(alpha=0.1, l1_ratio=0.1)
        net.fit(X_train, Y_train)
        y_pred_train = net.predict(X_train)
        y_pred_test = net.predict(X_test)
        
        mse_test = mean_squared_error(Y_test, y_pred_test)
        if mse_test < best_loss:
            best_loss = mse_test
            best_logistic_regression = net
            best_alpha = alpha
        
    net = best_logistic_regression

    mse_train = mean_squared_error(Y_train, y_pred_train)
    mse_test = mean_squared_error(Y_test, y_pred_test)
    r2_train = r2_score(Y_train, y_pred_train)
    r2_test = r2_score(Y_test, y_pred_test)
    res = {'mse_train': mse_train, 'mse_test': mse_test, 'r2_train': r2_train, 'r2_test': r2_test}
    
    res['best_alpha'] = best_alpha
    res['weights'] = net.coef_
    res['bias'] = net.intercept_
    return res



if __name__=='__main__':
    x = torch.randn(1000, 5)
    V = torch.randn(5)
    y = torch.matmul(x, V) > 0
    y = y.int()
    y[::2] = 2 # imbalanced classes
    print(y.sum())
    res = probe_classification(x, y, Normalize_X=False, fit_intercept=False, handle_imbalanced=True) 
    print(res['confusion_matrix_train'])
    print(res['confusion_matrix_test'])
    
    y_reg = torch.randn(1000)
    res = probe_regression(x, y_reg)
    print(res['mse_train'])
    
    
    