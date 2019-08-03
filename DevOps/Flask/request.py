'''
Sends data to the flask API and returns the predictions
'''
import requests
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Loading the data
iris_data = load_iris()
data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
data['Species'] = iris_data.target

# Assigning the features and labels
X = data.drop('Species', axis=1)  # Features
y = data['Species']  # Labels

# Specifying the URL for the flask app
url = 'http://localhost:5000/predict'

# JSONifying the data to be sent for the prediction
data_json = X.to_json(orient='split')

# Sending the request
r = requests.post(url, json=data_json)
print('Response code: {0}'.format(r.status_code))

# Formatting the returned predictions as a data frame
predictions = pd.read_json(r.json()['predictions'])
predictions
