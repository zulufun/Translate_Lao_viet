from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, DatasetDict
import torch
import os

def load_custom_dataset(lao_path='data/dev_lao.txt', vie_path='data/dev_viet.txt'):
    """
    Tải dữ liệu từ các file .txt
    
    :param lao_path: Đường dẫn file văn bản Lào
    :param vie_path: Đường dẫn file văn bản Việt
    :return: Dataset cho việc fine-tune
    """
    # Đọc dữ liệu từ file
    with open(lao_path, 'r', encoding='utf-8') as f:
        lao_lines = f.readlines()
    
    with open(vie_path, 'r', encoding='utf-8') as f:
        vie_lines = f.readlines()
    
    # Kiểm tra số lượng dòng
    assert len(lao_lines) == len(vie_lines), "Số lượng dòng không khớp giữa hai file"
    
    # Loại bỏ các dòng trắng
    lao_lines = [line.strip() for line in lao_lines if line.strip()]
    vie_lines = [line.strip() for line in vie_lines if line.strip()]
    
    # Tạo dataset
    dataset = Dataset.from_dict({
        'lao': lao_lines,
        'vie': vie_lines
    })
    
    # Chia dataset thành train và validation
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

def load_and_preprocess_data(dataset):
    """
    Tiền xử lý dữ liệu cho model
    
    :param dataset: Dataset gốc
    :return: Dataset đã được tokenize
    """
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
    """
    Fine-tune model dịch từ Lào sang Việt
    """
    # Tải dataset
    dataset = load_custom_dataset()
    
    # Tải model gốc
    model_name = "facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Tiền xử lý dữ liệu
    tokenized_dataset, tokenizer = load_and_preprocess_data(dataset)
    
    # Cấu hình training
    training_args = Seq2SeqTrainingArguments(
        output_dir="./lao_vie_translation_model",
        num_train_epochs=5,  # Tăng số epoch để học kỹ hơn
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    # Khởi tạo Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation']
    )
    
    # Bắt đầu huấn luyện
    trainer.train()
    
    # Lưu model
    trainer.save_model("./lao_vie_translation_model")
    
    # In thống kê
    print("Thống kê dữ liệu:")
    print(f"Tổng số mẫu: {len(dataset['train']) + len(dataset['validation'])}")
    print(f"Mẫu train: {len(dataset['train'])}")
    print(f"Mẫu validation: {len(dataset['validation'])}")

if __name__ == "__main__":
    fine_tune_model() 