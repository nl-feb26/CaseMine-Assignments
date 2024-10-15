import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Dataset processing
df = load_data()
X_train, X_test, y_train, y_test = train_test_split(df['tokenized'], df['label'], test_size=0.2, random_state=42)

# Training the BERT model
optimizer = AdamW(model.parameters(), lr=2e-5)

def train_model():
    model.train()
    for epoch in range(3):  # 3 epochs
        for idx, row in X_train.iterrows():
            inputs = torch.tensor([row['tokenized']])
            labels = torch.tensor([row['label']]).unsqueeze(0)
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}: Loss {loss.item()}")
    model.save_pretrained("models/bert_model")

if __name__ == "__main__":
    train_model()
