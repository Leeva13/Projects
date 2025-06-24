from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import whisper
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'ogg'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# Завантаження моделі Whisper (medium для кращої точності)
model = whisper.load_model("medium")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def split_lyrics(text):
    lines = text.split('\n')
    sections = []
    current_section = []
    section_type = "КУПЛЕТ"

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Визначення приспіву (наприклад, повторювані фрази)
        if line.count(' ') < 4 and (line.isupper() or 'приспів' in line.lower()):
            if current_section:
                sections.append(f"[{section_type}]\n" + '\n'.join(current_section))
                current_section = []
            section_type = "ПРИСПІВ"
        
        current_section.append(line)
    
    if current_section:
        sections.append(f"[{section_type}]\n" + '\n'.join(current_section))
    
    return sections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не вибрано'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не вибрано'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('process_file', filename=filename))
    return jsonify({'error': 'Недопустимий формат файлу'}), 400

@app.route('/process/<filename>')
def process_file(filename):
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        result = model.transcribe(audio_path, language='uk') # 'uk' для української
        recognized_text = result["text"].replace("  ", "\n") # Розділяємо рядки
        sections = split_lyrics(recognized_text)
    except Exception as e:
        recognized_text = f"Помилка: {str(e)}"
        sections = []
    return render_template('result.html', 
                         filename=filename,
                         sections=sections)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)