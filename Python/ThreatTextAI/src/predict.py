import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sys
import os

LABELS = ["attack", "movement", "disinfo", "neutral"]

# Ініціалізація токенізатора та моделі
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
if os.path.exists("models/best_model.pth"):
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=4)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu'), weights_only=True))
else:
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=4)  # Початкова модель
model.eval()

def predict(text: str):
    encoding = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(torch.device('cpu'))
    attention_mask = encoding['attention_mask'].to(torch.device('cpu'))
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).item()
        return LABELS[pred]

if __name__ == '__main__':
    sample_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Ворог обстріляв підстанцію"
    prediction = predict(sample_text)
    print(f"\n🔍 Текст: {sample_text}\n🧠 Клас: {prediction}")