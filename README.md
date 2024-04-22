# Text-Similarity-NLP
## Text Similarity using SentenceTransformer

This project aims to measure the similarity between pairs of texts by leveraging the power of the SentenceTransformer model. By preprocessing the text data and transforming it into numerical embeddings, the project computes the cosine similarity between the embeddings to determine the similarity score. This approach allows for efficient and accurate comparison of text pairs, making it useful for various natural language processing tasks.

## Installation

Clone the repository and install the required dependencies using `pip install -r requirements.txt`.

## Usage

Here dataset file which was used was named `DataNeuron_Text_Similarity.csv` in the Data directory and run the `main.py` script. The output will be saved to `OutputPartA.csv` in the Data directory.

## Requirements

- Python 3.x
- pandas
- nltk
- sentence-transformers
- torch
