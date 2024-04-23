# Text-Similarity-NLP
## Text Similarity using SentenceTransformer

This project aims to measure the similarity between pairs of texts by leveraging the power of the SentenceTransformer model. By preprocessing the text data and transforming it into numerical embeddings, the project computes the cosine similarity between the embeddings to determine the similarity score. This approach allows for efficient and accurate comparison of text pairs, making it useful for various natural language processing tasks. The final output is a CSV file containing the original text pairs, their preprocessed versions, and the calculated similarity scores.
For a detailed explanation of each step, please refer to the Project_Report.pdf

## Installation

Clone the repository and install the required dependencies using `pip install -r requirements.txt`.

## Project Files

1. **Project_Report.pdf**: This file contains a detailed report of the project, explaining each step in detail.

2. **app.py**: This is the main Flask application file. It loads the SentenceTransformer model, preprocesses the input text, and defines a route /similarity that accepts POST requests. When a request is received, it calculates the semantic textual similarity between the two input texts and returns the similarity score in response.

3. **test_api.py**: This file is used to test the Flask API. It sends a POST request to the /similarity route of our Flask application and prints the request and response bodies.

4. **text_similarity.py**: This Python script contains the code for Part A of the project, which involves preprocessing the text data, transforming it into numerical embeddings using the SentenceTransformer model, and computing the cosine similarity between pairs of texts.

5. **requirements.txt**: This file lists all the Python libraries that are required to run the project.
 
6. Data directory contains both input and output CSV files, whereas Other_Files directory contains cloud deployment files.

   
https://miro.medium.com/v2/resize:fit:2000/1*GXMlZw_Z217UIrUl5VfNIw.png

