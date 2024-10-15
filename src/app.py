from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, GPT2LMHeadModel, BertTokenizer, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load models
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.json
    text = data['text']
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = bert_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return jsonify({'prediction': 'positive' if prediction == 1 else 'negative'})

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data['prompt']
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=100)
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
