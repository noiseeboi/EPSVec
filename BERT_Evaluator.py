from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset
import numpy as np
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer


from datasets import Dataset, DatasetDict

def train_bert(train_dataset, seed, epochs=2):
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    num_labels = len(set(train_dataset['train']['label']))
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)
    
    train_tokenized = train_dataset.map(tokenize, batched=True, batch_size=None)


    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    batch_size = 16
    epochs = epochs
    learning_rate = 5e-5
    training_args = TrainingArguments(
        output_dir="./BERTcheckpoints",          
        num_train_epochs=epochs,             
        per_device_train_batch_size=batch_size,   
        per_device_eval_batch_size=batch_size,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir="./logs",        
        report_to="none",
        seed = seed,
        learning_rate=learning_rate,
    )

    trainer = Trainer(
        model=model,                        
        args=training_args,                 
        train_dataset=train_tokenized["train"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer, tokenizer

def evaluate_bert(trainer, tokenizer, test_set):
    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)    
    encoded = test_set.map(tokenize, batched=True, batch_size=None)
    eval_results = trainer.evaluate(eval_dataset=encoded["test"])
    # print(f"Eval results: {eval_results}")    
    return eval_results


def make_datasets(dataset, split = 'train'):
    all_text = []
    all_labels = []
    for key in dataset.keys():
        assert isinstance(key, int), 'key is not int'
        all_labels.extend([key for _ in dataset[key]])
        all_text.extend(dataset[key])
    
    train_data = {
        "text": all_text,
        "label": all_labels
    }
    train_dataset = Dataset.from_dict(train_data)
    dataset = DatasetDict({
        split: train_dataset,
    })
    return dataset

class BERTEvaluator:
    def __init__(self, real_train_data, real_test_data, epochs=2):
        self.real_train_set = make_datasets(real_train_data, 'train')
        self.real_test_set  = make_datasets(real_test_data, 'test')
        self.epochs = epochs
        # self.real_eval_acc = evaluate_bert(self.real_bert, self.real_tokenizer, self.real_test_set)['eval_accuracy']
        # print(real_eval)
    
    def evaluate(self, synthetic_data, seed):
        synthetic_train_set = make_datasets(synthetic_data, 'train')
        synthetic_test_set = make_datasets(synthetic_data, 'test')
        
        real_bert, real_tokenizer = train_bert(self.real_train_set, seed, epochs=self.epochs)
        synthetic_bert, synthetic_tokenizer = train_bert(synthetic_train_set, seed, epochs=self.epochs)
        TRTS = evaluate_bert(real_bert, real_tokenizer, synthetic_test_set)['eval_accuracy']
        TSTR = evaluate_bert(synthetic_bert, synthetic_tokenizer, self.real_test_set)['eval_accuracy']

        return {'TRTS': TRTS, 'TSTR': TSTR}


