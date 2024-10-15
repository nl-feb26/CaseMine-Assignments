import pandas as pd
import re
from pymongo import MongoClient
from transformers import BertTokenizer

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['nlp_project']
collection = db['imdb_data']

def preprocess_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

# Load and preprocess the IMDB dataset
def load_data():
    df = pd.read_csv(r"D:\Projects\Datasets\IMDB Dataset.csv\IMDB Dataset.csv")
    
    # Sampling 50K rows (optional, you can use all data if desired)
    df = df.sample(n=50000, random_state=42)

    # Preprocess the review text
    df['cleaned_text'] = df['review'].apply(preprocess_text)

    # Tokenize using BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df['tokenized'] = df['cleaned_text'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length'))

    # Insert into MongoDB
    collection.insert_many(df.to_dict('records'))
    
    return df

if __name__ == "__main__":
    load_data()
