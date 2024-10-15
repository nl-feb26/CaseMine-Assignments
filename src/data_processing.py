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

# Load IMDB dataset
def load_data():
    df = pd.read_csv('https://datasets.imdbws.com/title.basics.tsv.gz', delimiter='\t', low_memory=False)
    df = df[['primaryTitle', 'isAdult', 'startYear']].dropna()  # Dropping unnecessary columns
    df = df.sample(n=50000, random_state=42)  # Sampling 50K rows

    df['cleaned_text'] = df['primaryTitle'].apply(preprocess_text)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df['tokenized'] = df['cleaned_text'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length'))

    collection.insert_many(df.to_dict('records'))  # Load into MongoDB
    return df

if __name__ == "__main__":
    load_data()
