from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import logging
from functools import lru_cache
import os

app = Flask(__name__)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageTranslator:
    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Tải model và tokenizer"""
        try:
            logger.info(f"Đang tải model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model đã được tải thành công!")
        except Exception as e:
            logger.error(f"Lỗi khi tải model: {e}")
            raise e
    
    @lru_cache(maxsize=100)
    def translate(self, text, source_lang='lao', target_lang='vie', max_length=512):
        """Dịch text giữa các ngôn ngữ"""
        if not text or not text.strip():
            return ""
        
        try:
            # Mã ngôn ngữ theo chuẩn NLLB
            source_code = 'lao_Laoo'
            target_code = 'vie_Latn'
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[target_code],
                    max_length=max_length,
                    num_beams=4,
                    length_penalty=0.6,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode kết quả
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated_text.strip()
            
        except Exception as e:
            logger.error(f"Lỗi khi dịch: {e}")
            return f"Lỗi: Không thể dịch văn bản này"

# Khởi tạo translator
translator = LanguageTranslator()

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    """API endpoint để dịch text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        source_lang = data.get('source_lang', 'lao')
        target_lang = data.get('target_lang', 'vie')
        
        if not text:
            return jsonify({'error': 'Vui lòng nhập văn bản cần dịch'}), 400
        
        # Kiểm tra độ dài
        if len(text) > 1000:
            return jsonify({'error': 'Văn bản quá dài (tối đa 1000 ký tự)'}), 400
        
        # Dịch
        translated = translator.translate(text, source_lang, target_lang)
        
        return jsonify({
            'original': text,
            'translated': translated,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Lỗi API: {e}")
        return jsonify({'error': 'Có lỗi xảy ra khi dịch'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': translator.model is not None,
        'device': str(translator.device)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)