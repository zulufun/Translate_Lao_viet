from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import torch

def load_and_preprocess_data():
    # Thay thế bằng dataset của bạn
    dataset = load_dataset('opus_books', 'lao-vie')
    
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        inputs = [ex for ex in examples['lao']]
        targets = [ex for ex in examples['vie']]
        
        model_inputs = tokenizer(
            inputs, 
            max_length=512, 
            truncation=True, 
            padding='max_length'
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=512, 
                truncation=True, 
                padding='max_length'
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset['train'].column_names
    )
    
    return tokenized_dataset, tokenizer

def fine_tune_model():
    model_name = "facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    tokenized_dataset, tokenizer = load_and_preprocess_data()
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./lao_vie_translation_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation']
    )
    
    trainer.train()
    trainer.save_model("./lao_vie_translation_model")

if __name__ == "__main__":
    fine_tune_model() 