# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors

import os
import sys
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

app = Flask(__name__)
CORS(app)  # Apply CORS to your app

@app.route('/api/openai', methods=['POST'])
def openai():
    data = request.get_json()
    if 'query' in data:
        query = data['query']

        # Process the query
        loader = TextLoader("data/Usedata.txt")
        index = VectorstoreIndexCreator().from_loaders([loader])
        response = index.query(query, llm=ChatOpenAI(model="gpt-3.5-turbo"))

        return jsonify({'response': response})
    else:
        return jsonify({'error': 'Invalid request format. Please provide a query.'}), 400
    
@app.route('/api/openai_custom', methods=['POST'])
def openai_custom():
    data = request.get_json()
    if 'query' in data:
        query = data['query']

        # Process the query
        loader = TextLoader("data/Usedata.txt")
        index = VectorstoreIndexCreator().from_loaders([loader])
        response = index.query(query)

        return jsonify({'response': response})
    else:
        return jsonify({'error': 'Invalid request format. Please provide a query.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)