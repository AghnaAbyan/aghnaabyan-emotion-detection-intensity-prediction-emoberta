import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
from textblob import TextBlob
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
nltk.download('averaged_perceptron_tagger_eng')

class Config:
    MODEL_NAME = "tae898/emoberta-large"
    MAX_LEN = 128
    VALID_BATCH_SIZE = 16
    NUM_LABELS = 5
    EMOTION_NAMES = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmotionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def get_features(self, text):
        blob = TextBlob(text)
        words = text.split()
        return {
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity,
            'text_length': len(text),
            'word_count': len(words),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1),
            'exclamation_count': text.count('!') / (len(words) + 1),
            'question_count': text.count('?') / (len(words) + 1)
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        features = self.get_features(text)
        feature_tensor = torch.tensor([
            features['sentiment_polarity'],
            features['sentiment_subjectivity'],
            features['text_length'] / 1000,
            features['word_count'] / 100,
            features['uppercase_ratio'],
            features['exclamation_count'],
            features['question_count']
        ], dtype=torch.float)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': feature_tensor
        }

class MultiLabelEmotionModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.modelUsed = AutoModel.from_pretrained(model_name)

        for param in self.modelUsed.embeddings.parameters():
            param.requires_grad = False
        for i in range(6):
            for param in self.modelUsed.encoder.layer[i].parameters():
                    param.requires_grad = False

        hidden_size = self.modelUsed.config.hidden_size

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        self.feature_processor = nn.Sequential(
            nn.Linear(7, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask, features, labels=None):
        outputs = self.modelUsed(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        attention_weights = self.attention(hidden_states)
        attended_output = torch.sum(attention_weights * hidden_states, dim=1)

        processed_features = self.feature_processor(features)
        combined_features = torch.cat([attended_output, processed_features], dim=1)
        logits = self.classifier(combined_features)
        loss = None

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

def predict_track_a(texts, model_path, output_csv=None):
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    dataset = EmotionDataset(texts=texts, tokenizer=tokenizer, max_len=config.MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=config.VALID_BATCH_SIZE)

    model = MultiLabelEmotionModel(config.MODEL_NAME, config.NUM_LABELS)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            features = batch['features'].to(config.DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            logits_tensor = outputs['logits']
            probs = torch.sigmoid(logits_tensor).cpu().numpy()
            all_preds.append(probs)

    all_preds = np.vstack(all_preds)

    print("\n Raw predictions:\n")
    print(pd.DataFrame(all_preds, columns=config.EMOTION_NAMES))

    binary_preds = (all_preds >= 0.5).astype(int)

    results_df = pd.DataFrame(binary_preds, columns=config.EMOTION_NAMES)
    results_df['text'] = texts

    print("\n Final track a predictions:\n")
    print(results_df)

    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"\n Saved to {output_csv}")

    return results_df

if __name__ == "__main__":
    # .pt
    MODEL_PATH = "best_model_final_track_a.pt"

    # Text
    texts_to_predict = [
        """Been coughing my lungs out for a week now = ( Hopefully I get better soon enough because Chinese New Year is coming and I am craving to eat mandarin oranges!!!"""
    ]

    predict_track_a(texts_to_predict, MODEL_PATH, output_csv="track_a_predictions.csv")