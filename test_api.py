import requests
import json

def test_api():
    # Define the URL of your API
    url = 'http://127.0.0.1:5001/similarity'  
# Add /similarity to the URL

    # Define a test instance
    test_instance = {
        'text1': 'This is the first piece of text.',
        'text2': 'This is the second piece of text.'
    }

    # Send a POST request to the API
    response = requests.post(url, json=test_instance)

    # Check the status code of the response
    assert response.status_code == 200

    # Parse the response
    similarity_score = response.json()['similarity score']

    # Print the similarity score
    print('Request body:', json.dumps(test_instance))
    print('Response body:', json.dumps({'similarity score': similarity_score}))

if __name__ == '__main__':
    test_api()
