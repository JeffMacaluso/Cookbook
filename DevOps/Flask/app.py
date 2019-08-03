'''
The Flask app that loads the model and generates predictions
'''
import os
import sys
import json
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pickle

# Initializing the flask app
app = Flask(__name__)

# Loading the model
os.chdir(sys.path[0])  # Setting the directory to the same directory as the script
model_file_path = './models/model.pkl'
model = pickle.load(open(model_file_path, 'rb'))
print('Model Loaded')

# Name of the species to return
class_dict = {0:'Setosa', 1:'Versicolour', 2: 'Virginica'}


@app.route('/test/',methods=['GET','POST'])
def test():
    responses = jsonify(predictions=json.dumps('The flask app is working'))
    responses.status_code = 200
    return responses

@app.route('/predict', methods=['POST'])
def predict():
    '''
    API call to make predictions on received data

    Receives a pandas dataframe that was sent as a payload and generates batch predictions
    '''
    try:
        json_data = request.get_json()
        data = pd.read_json(json_data, orient='split')
        
        # TODO: Put any preprocessing steps here

    except Exception as e:
        raise e
    
    if data.empty:
        return bad_request()
    
    else:
        # Creating predictions
        print('Creating predictions for {0} records'.format(data.shape[0]))
        
        class_probabilities = model.predict_proba(data)
        predictions = np.argmax(class_probabilities, axis=1)
        predictions = [class_dict[prediction] for prediction in predictions]
        predictions = pd.DataFrame({'prediction': predictions,
                                    'probability_setosa': class_probabilities[:, 0],
                                    'probability_versicolour': class_probabilities[:, 1],
                                    'probability_virginica': class_probabilities[:, 2]})
        predictions['index'] = range(predictions.shape[0])
        
        # Returning the responses
        responses = jsonify(predictions=predictions.to_json(orient='records'))
        responses.status_code = 200
        
        return responses


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='localhost', port=port, debug=True)
