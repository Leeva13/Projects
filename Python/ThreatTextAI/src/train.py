import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
from dataset import ThreatDataset
from sklearn.metrics import f1_score

def train_model():
    # Налаштування
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    PATIENCE = 3

    print("DEBUG: Початок завантаження даних")
    # Завантаження даних
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    dataset = ThreatDataset('data/train.csv', tokenizer=tokenizer)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_idx, val_idx = torch.utils.data.random_split(range(len(dataset)), [train_size, val_size])

    print(f"DEBUG: Дані завантажено. Train size: {train_size}, Val size: {val_size}")
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx.indices))
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_idx.indices))

    # Обчислення ваг класів
    df = pd.read_csv('data/train.csv')
    class_counts = df['label'].value_counts().sort_index().values
    print(f"DEBUG: Кількість класів: {class_counts}")
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Ініціалізація моделі
    print("DEBUG: Ініціалізація моделі")
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=4).to(device)
    # Розморожуємо верхні 2 шари BERT
    for param in model.bert.encoder.layer[-2:].parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # Тренування
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"DEBUG: Початок епохи {epoch+1}")
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for i, batch in enumerate(train_loader):
            print(f"DEBUG: Обробка батчу {i+1}/{len(train_loader)}")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        
        # Валідація
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                print(f"DEBUG: Валідація батчу {i+1}/{len(val_loader)}")
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f'Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% - Val F1: {val_f1:.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("✅ Збережено нову найкращу модель!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Рання зупинка: валідаційна втрата не покращується.")
                break

    print("Тренування завершено!")

def predict_text(text, model, tokenizer):
    model.eval()
    encoding = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**encoding)
        _, predicted = torch.max(outputs.logits, 1)
    return predicted.item()

if __name__ == "__main__":
    train_model()