import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the processed IMDB data
df = pd.read_csv('processed_imdb_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['tokenized'], df['sentiment'], test_size=0.2, random_state=42)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Optimizer setup
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training function
def train_model():
    model.train()
    for epoch in range(3):  # Train for 3 epochs
        total_loss = 0  # Initialize total loss for monitoring
        for idx in range(len(X_train)):
            # Convert the tokenized input (string) back to a list of integers
            tokenized_input = eval(X_train.iloc[idx])  # Convert string representation back to a list

            # Create input tensor and attention mask
            inputs = torch.tensor(tokenized_input).unsqueeze(0)  # Add batch dimension
            attention_mask = (inputs != 0).long()  # Create attention mask (1 for real tokens, 0 for padding)
            labels = torch.tensor([y_train.iloc[idx]]).unsqueeze(0)  # Sentiment label

            optimizer.zero_grad()
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Accumulate loss

            # Print loss every 10 batches
            if idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {idx}, Loss: {loss.item()}")

        avg_loss = total_loss / len(X_train)  # Average loss for the epoch
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss}")

    # Save the trained model
    model.save_pretrained("models/bert_model")

if __name__ == "__main__":
    train_model()
