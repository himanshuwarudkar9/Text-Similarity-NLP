
from google.colab import files
uploaded = files.upload()

"""# Importing the required libraries"""

import pandas as pd
import io
from nltk.corpus import wordnet
import autocorrect
from autocorrect import Speller
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

"""# Reading the file"""

df = pd.read_csv(io.BytesIO(uploaded["DataNeuron_Text_Similarity.csv"]))

"""# Preprocssing the file"""

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



"""# Transforming the data based on above pre-processing steps"""

df['text1_processed'] = df['text1'].apply(preprocess_text)
df['text2_processed'] = df['text2'].apply(preprocess_text)

"""# Applying the SentenceTransformer model"""

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-MiniLM-L6-v2') #all-MiniLM-L12-v2 #all-MiniLM-L12-v2 #all-MiniLM-L6-v2
#model = SentenceTransformer('all-MiniLM-L12-v2')
# Encode sentences into embeddings
embeddings1 = model.encode(df['text1_processed'], convert_to_tensor=True)
embeddings2 = model.encode(df['text2_processed'], convert_to_tensor=True)

# Calculate the cosine similarity between the embeddings
similarity_scores = util.cos_sim(embeddings1, embeddings2)
average_similarity = similarity_scores.cpu().diagonal().numpy()
df['similarity'] = average_similarity
df.head()

# Saving dataframe into a csv format
file_path = "OutputPartA.csv"

df.to_csv(file_path, index=False)


