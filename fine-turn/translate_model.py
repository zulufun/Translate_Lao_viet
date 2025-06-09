from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class TranslationModel:
    def __init__(self, model_path="./lao_vie_translation_model"):
        """
        Khởi tạo model dịch từ Lào sang Việt
        
        :param model_path: Đường dẫn tới model đã fine-tune
        """
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    def translate(self, text):
        """
        Dịch văn bản từ Lào sang Việt
        
        :param text: Văn bản nguồn bằng tiếng Lào
        :return: Văn bản dịch sang tiếng Việt
        """
        # Mã hóa văn bản đầu vào
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True, 
            padding=True
        )
        
        # Sinh ra văn bản dịch
        outputs = self.model.generate(
            **inputs, 
            max_length=512, 
            num_beams=4, 
            early_stopping=True
        )
        
        # Giải mã văn bản dịch
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translated_text

def main():
    # Ví dụ sử dụng model
    translator = TranslationModel()
    
    # Thử dịch một văn bản mẫu
    sample_text = "ສະບາຍດີ, ຂ້ອຍຊື່ວ່າ..."
    translated = translator.translate(sample_text)
    
    print("Văn bản gốc (Lào):", sample_text)
    print("Văn bản dịch (Việt):", translated)

if __name__ == "__main__":
    main() 