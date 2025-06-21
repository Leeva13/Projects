import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from preprocess import preprocess_text

LABEL2IDX = {"attack": 0, "movement": 1, "disinfo": 2, "neutral": 3}

class ThreatDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_path)
        print(f"DEBUG: Завантажено {len(self.df)} рядків із {csv_path}")
        self.texts = []
        self.labels = []
        for text, label in zip(self.df['text'], self.df['label']):
            augmented_texts = preprocess_text(text)
            self.texts.extend(augmented_texts)
            self.labels.extend([LABEL2IDX[label]] * len(augmented_texts))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

def get_dataloader(csv_path, tokenizer, batch_size=16):
    dataset = ThreatDataset(csv_path, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    loader = get_dataloader("data/train.csv", tokenizer)
    for batch in loader:
        print("Batch input_ids shape:", batch['input_ids'].shape)
        print("Batch attention_mask shape:", batch['attention_mask'].shape)
        print("Batch labels:", batch['labels'])
        break