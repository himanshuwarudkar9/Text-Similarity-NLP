from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import io
from nltk.corpus import wordnet
import autocorrect
from autocorrect import Speller
import re
import numpy as np
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Data preprocessing
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
   

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

@app.route('/similarity', methods=['POST'])
def similarity():
    # Get the texts from the request
    text1 = request.json['text1']
    text2 = request.json['text2']

    # Preprocess the texts
    text1_processed = preprocess_text(text1)
    text2_processed = preprocess_text(text2)

    # Convert the texts to embeddings
    embedding1 = model.encode([text1_processed], convert_to_tensor=True)
    embedding2 = model.encode([text2_processed], convert_to_tensor=True)

    # Calculate the semantic textual similarity
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2)

    # Return the similarity score
    return jsonify({'similarity score': similarity_score.item()})

if __name__ == '__main__':
    app.run(debug=True,port=5001)
