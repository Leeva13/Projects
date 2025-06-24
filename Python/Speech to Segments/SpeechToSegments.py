import numpy as np
import librosa
import soundfile as sf
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

def process_audio_file(file_path, output_dir='output_letters', min_silence_len=300, silence_thresh=-40, keep_silence=100):
    """
    Обробляє аудіофайл, розділяючи його на окремі частини з вимовою літер.
    
    Параметри:
    - file_path: шлях до MP3 файлу
    - output_dir: директорія для збереження результатів
    - min_silence_len: мінімальна довжина тиші для розділення (мс)
    - silence_thresh: поріг гучності для визначення тиші (дБ)
    - keep_silence: скільки тиші зберігати до і після звуку (мс)
    """
    # Створюємо директорію для результатів, якщо вона не існує
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Завантажуємо аудіо
    print(f"Завантаження файлу: {file_path}")
    sound = AudioSegment.from_file(file_path, format="mp3")
    
    # Розділяємо на частини за тишею
    print("Розділення аудіо на частини...")
    audio_chunks = split_on_silence(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    
    print(f"Знайдено {len(audio_chunks)} аудіо частин")
    
    # Зберігаємо кожен фрагмент
    for i, chunk in enumerate(audio_chunks):
        # Формуємо ім'я файлу
        output_file = os.path.join(output_dir, f"letter_{i+1}.wav")
        print(f"Збереження: {output_file}")
        chunk.export(output_file, format="wav")
    
    print("Обробка завершена!")
    return len(audio_chunks)

def analyze_chunks(output_dir='output_letters'):
    """
    Аналізує створені аудіо частини і виводить інформацію про них.
    """
    chunks = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    chunks.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    print(f"\nАналіз {len(chunks)} аудіо фрагментів:")
    for chunk_file in chunks:
        file_path = os.path.join(output_dir, chunk_file)
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        print(f"{chunk_file}: тривалість {duration_ms} мс, гучність {audio.dBFS:.1f} дБ")

def fine_tune_parameters(file_path):
    """
    Інтерактивна функція для знаходження оптимальних параметрів розділення.
    """
    min_silence_len = 300  # початкові значення
    silence_thresh = -40
    keep_silence = 100
    
    while True:
        print(f"\nПоточні параметри:")
        print(f"- min_silence_len: {min_silence_len} мс")
        print(f"- silence_thresh: {silence_thresh} дБ")
        print(f"- keep_silence: {keep_silence} мс")
        
        chunks_count = process_audio_file(
            file_path, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        analyze_chunks()
        
        print("\nЩо хочете змінити?")
        print("1. min_silence_len (для більших значень - менше частин)")
        print("2. silence_thresh (нижче значення робить поріг тиші більш строгим)")
        print("3. keep_silence (скільки тиші зберігати навколо звуку)")
        print("4. Завершити налаштування")
        
        choice = input("Ваш вибір (1-4): ")
        
        if choice == '1':
            min_silence_len = int(input("Введіть нове значення min_silence_len (мс): "))
        elif choice == '2':
            silence_thresh = int(input("Введіть нове значення silence_thresh (дБ): "))
        elif choice == '3':
            keep_silence = int(input("Введіть нове значення keep_silence (мс): "))
        elif choice == '4':
            break
        else:
            print("Невірний вибір. Спробуйте знову.")
    
    return min_silence_len, silence_thresh, keep_silence

# Приклад використання:
if __name__ == "__main__":
    # Шлях до вашого файлу
    file_path = input("Введіть шлях до MP3 файлу: ")
    
    print("\nОберіть режим роботи:")
    print("1. Автоматичне розділення з параметрами за замовчуванням")
    print("2. Налаштування параметрів розділення")
    
    mode = input("Ваш вибір (1-2): ")
    
    if mode == '1':
        # Використовуємо параметри за замовчуванням
        process_audio_file(file_path)
        analyze_chunks()
    elif mode == '2':
        # Налаштування параметрів
        fine_tune_parameters(file_path)
    else:
        print("Невірний вибір. Використовуємо автоматичний режим.")
        process_audio_file(file_path)
        analyze_chunks()