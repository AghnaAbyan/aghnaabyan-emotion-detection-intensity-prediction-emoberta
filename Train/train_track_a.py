import nlpaug
import nlpaug.augmenter.word as naw
import nltk
import numpy as np
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import warnings
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
warnings.filterwarnings('ignore')
nltk.download('averaged_perceptron_tagger_eng')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

waste_name = '/home/asrock/QNLP/abyanproject/Waste/track_a/08-07/8-7-2025_6.csv'
use_name = '/home/asrock/QNLP/abyanproject/Use/track_a/08-07/8-7-2025_6.csv'

class Config:
  MODEL_NAME = "roberta-large"
  MAX_LEN = 128
  TRAIN_BATCH_SIZE = 16
  VALID_BATCH_SIZE = 16
  EPOCHS = 5
  LEARNING_RATE = 2e-5
  WARMUP_PROPORTION = 0.1
  NUM_LABELS = 5
  EMOTION_NAMES = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]
  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  THRESHOLD = 0.5  # Binary threshold

class EmotionDataset(Dataset):
  def __init__(self, texts, labels=None, tokenizer=None, max_len=128, augment=False):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.augment = augment

    if augment:
      self.aug_syn = naw.SynonymAug(aug_p=0.3)
      self.aug_context = naw.ContextualWordEmbsAug(
        model_path=config.MODEL_NAME,
        action="substitute"
      )

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

    if self.augment and self.labels is not None: # Augmentation
      if random.random() < 0.3:
        aug_type = random.choice(['synonym', 'contextual'])
        if aug_type == 'synonym':
          text = self.aug_syn.augment(text)[0]
        else:
          text = self.aug_context.augment(text)[0]

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

    # Tokenization
    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt'
    )

    item = {
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'features': feature_tensor
    }

    if self.labels is not None:
      item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)

    return item

class MultiLabelEmotionModel(nn.Module):
  def __init__(self, model_name, num_labels):
    super().__init__()
    self.modelUsed = AutoModel.from_pretrained(model_name)

    # Transfer learning
    for param in self.modelUsed.embeddings.parameters():
      param.requires_grad = False
    for i in range(6):
      for param in self.modelUsed.encoder.layer[i].parameters():
        param.requires_grad = False

    hidden_size = self.modelUsed.config.hidden_size

    # Self-Attention
    self.attention = nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.Tanh(),
      nn.Linear(hidden_size, 1),
      nn.Softmax(dim=1)
    )

    # Feature processing
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

    # Classifier
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

    logits = self.classifier(combined_features) # Sigmoid for binary
    loss = None
    if labels is not None:
      # BCE
      loss_fct = nn.BCEWithLogitsLoss()
      loss = loss_fct(logits, labels)

    return {"loss": loss, "logits": logits}
  
def train_model(config, train_data, val_data=None):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_dataset = EmotionDataset(
        texts=train_data['text'].values,
        labels=train_data[config.EMOTION_NAMES].values,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN,
        augment=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True
    )

    val_loader = None
    if val_data is not None:
        has_labels = not val_data[config.EMOTION_NAMES].isna().all().all()
        
        if has_labels:
            val_dataset = EmotionDataset(
                texts=val_data['text'].values,
                labels=val_data[config.EMOTION_NAMES].values,
                tokenizer=tokenizer,
                max_len=config.MAX_LEN
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.VALID_BATCH_SIZE
            )
        else:
            print("Validation data has no labels")

    model = MultiLabelEmotionModel(config.MODEL_NAME, config.NUM_LABELS)
    model = model.to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_PROPORTION)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_f1 = 0
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        model.train()
        train_losses = []

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            features = batch['features'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                features=features,
                labels=labels
            )

            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        print(f"Training loss: {avg_train_loss:.4f}")

        if val_loader is not None:
            model.eval()
            val_losses = []
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(config.DEVICE)
                    attention_mask = batch['attention_mask'].to(config.DEVICE)
                    features = batch['features'].to(config.DEVICE)
                    labels = batch['labels'].to(config.DEVICE)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        features=features,
                        labels=labels
                    )

                    loss = outputs['loss']
                    val_losses.append(loss.item())

                    logits = outputs['logits']
                    preds = torch.sigmoid(logits) >= config.THRESHOLD

                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

            avg_val_loss = np.mean(val_losses)
            print(f"Validation loss: {avg_val_loss:.4f}")

            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)

            # Calculate Macro F1-Score across all emotions
            macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            print(f"Macro F1: {macro_f1:.4f}")

            if macro_f1 > best_val_f1:
                best_val_f1 = macro_f1
                torch.save(model.state_dict(), 'model_track_a.pt')
                print("Saved")
        else:
            # Save model for each epoch if no val is available
            torch.save(model.state_dict(), f'model_track_a_epoch_{epoch+1}.pt')
            print(f"Saved model for epoch {epoch+1}")

    return model, tokenizer

def predict(texts, model, tokenizer, config):
  dataset = EmotionDataset(
    texts=texts,
    tokenizer=tokenizer,
    max_len=config.MAX_LEN
  )

  dataloader = DataLoader(dataset, batch_size=config.VALID_BATCH_SIZE)
  model.eval()
  predictions = []

  with torch.no_grad():
    for batch in dataloader:
      input_ids = batch['input_ids'].to(config.DEVICE)
      attention_mask = batch['attention_mask'].to(config.DEVICE)
      features = batch['features'].to(config.DEVICE)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        features=features
      )

      logits = outputs['logits']
      probs = torch.sigmoid(logits)
      preds = (probs >= config.THRESHOLD).int()
      predictions.append(preds.cpu().numpy())

  return np.vstack(predictions)

if __name__ == "__main__":
    start_time = time.time()
    set_seed(42)
    # Load data
    train_data = pd.read_csv('/home/asrock/QNLP/abyanproject/Track_A_Train_Dev_Combined.csv')
    test_data = pd.read_csv('/home/asrock/QNLP/abyanproject/Test_Eng_A.csv')
    
    # Split training data into train and validation
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    
    # Initialize configuration
    config = Config()
    
    # Train model with the validation split from training data
    model, tokenizer = train_model(config, train_data, val_data)
    
    # Make predictions on test data
    test_predictions = predict(test_data['text'].values, model, tokenizer, config)
    
    # Save predictions
    test_results = test_data.copy()
    test_results[Config.EMOTION_NAMES] = test_predictions
    test_results.to_csv(waste_name, index=False)
    
    # Create submission file without text column
    df_operation = test_results.drop(['text'], axis=1)
    output_path = use_name
    df_operation.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    end_time = time.time()  # End timer
    total_time = end_time - start_time
    print(f"\n Total time taken: {total_time:.2f} seconds") # Time evaluation