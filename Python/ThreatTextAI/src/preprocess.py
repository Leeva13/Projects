# preprocess.py (створи цей файл, якщо його немає)
import random
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_p=0.3)  # Аугментація синонімами

def preprocess_text(text):
    augmented_texts = [aug.augment(text)[0] for _ in range(2)]  # Генеруємо 2 варіанти
    return [text] + augmented_texts  # Повертаємо оригінал + варіанти