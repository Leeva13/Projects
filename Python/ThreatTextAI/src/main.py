import sys
import os
from train import train_model, predict_text
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch
from dataset import ThreatDataset
from sklearn.metrics import accuracy_score, f1_score

def evaluate_test_data():
    if not os.path.exists("models/best_model.pth"):
        print("\n⚠️ Помилка: Модель ще не натренована. Спочатку виберіть опцію 1 для тренування.")
        return
    
    # Завантаження моделі
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=4)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    model.to(torch.device('cpu'))  # Переміщаємо на CPU для сумісності

    # Завантаження тестових даних
    if not os.path.exists("data/test.csv"):
        print("\n⚠️ Помилка: Файл data/test.csv не знайдено. Створіть його з колонками 'text' і 'label'.")
        return

    test_dataset = ThreatDataset('data/test.csv', tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Оцінка
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(torch.device('cpu'))
            attention_mask = batch['attention_mask'].to(torch.device('cpu'))
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    total = len(all_labels)

    print("Деталі передбачень:")
    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
        print(f"Приклад {i+1}: Передбачено {pred} (справжнє {label})")

    print(f"\n📊 Результати тестування на {total} прикладах:")
    print(f"✅ Правильно вгадано: {correct} ({accuracy*100:.2f}%)")
    print(f"📈 F1-Score: {f1:.4f}")

def main():
    """Основна функція для запуску ThreatTextAI через термінал."""
    print("Ласкаво просимо до ThreatTextAI!")
    print("Цей проєкт класифікує текстові повідомлення за типом загроз.")
    print("DEBUG: Дійшов до початку циклу")
    
    # Ініціалізація моделі для прогнозів
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=4)
    if os.path.exists("models/best_model.pth"):
        model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
        model.eval()
    
    while True:
        print("\n=== Меню ===")
        print("1. Натренувати модель")
        print("2. Зробити передбачення")
        print("3. Перевірити модель на тестових даних")
        print("4. Вийти")
        choice = input("Вибери опцію (1-4): ")

        if choice == "1":
            print("\nПочинаємо тренування моделі...")
            train_model()
            print("Тренування завершено!")
            # Оновлюємо модель після тренування
            model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
            model.eval()
        elif choice == "2":
            if not os.path.exists("models/best_model.pth"):
                print("\n⚠️ Помилка: Модель ще не натренована. Спочатку виберіть опцію 1 для тренування.")
            else:
                text = input("\nВведи текст для класифікації: ")
                label = predict_text(text, model, tokenizer)
                print(f"\n📝 Текст: {text}")
                print(f"➡️ Клас: {label}")
        elif choice == "3":
            evaluate_test_data()
        elif choice == "4":
            print("Дякую за використання ThreatTextAI. До зустрічі!")
            sys.exit()
        else:
            print("Невірний вибір. Спробуй ще раз.")

if __name__ == "__main__":
    main()