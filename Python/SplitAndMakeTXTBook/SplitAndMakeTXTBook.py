import os
from pathlib import Path
from PyPDF2 import PdfReader

# === Налаштування ===
input_path = "Pro_Viyny.txt"  # Замінити на свій файл

output_dir = "splitted_output"
os.makedirs(output_dir, exist_ok=True)

CHUNK_SIZE = 20000

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def save_chunks(text, base_filename):
    total_parts = (len(text) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(total_parts):
        chunk = text[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        part_filename = f"{base_filename}_part_{i+1}.txt"
        part_path = os.path.join(output_dir, part_filename)
        with open(part_path, "w", encoding="utf-8") as f:
            f.write(chunk)
        print(f"Збережено: {part_path}")

def convert_and_split_file(input_path):
    ext = Path(input_path).suffix.lower()
    base_name = Path(input_path).stem

    if ext == ".pdf":
        text = extract_text_from_pdf(input_path)
    elif ext == ".txt":
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(input_path, "r", encoding="windows-1251") as f:
                text = f.read()
    else:
        raise ValueError("Підтримуються лише .pdf або .txt")

    save_chunks(text, base_name)

# === Виконання ===
convert_and_split_file(input_path)
