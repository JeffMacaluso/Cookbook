'''
Trains the machine learning model to be operationalized with Flask
'''
import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Loading the data
iris_data = load_iris()
data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
data['Species'] = iris_data.target

# Assigning the features and labels
X = data.drop('Species', axis=1)  # Features
y = data['Species']  # Labels

# Splitting between training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=46)

# Training the model
model = LogisticRegression(solver='lbfgs', max_iter=300, multi_class='auto')
print('Training the model')
model.fit(X_train, y_train)

# Testing the model
accuracy = model.score(X_test, y_test)
print('Model Accuracy: {0}'.format(accuracy))

# Exporting the model
os.chdir(sys.path[0])  # Setting the directory to the same directory as the script
file_name = './models/model.pkl'
pickle.dump(model, file=open(file_name, 'wb'))
print('Model exported')
