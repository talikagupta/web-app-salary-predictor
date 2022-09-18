# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset into a pandas dataframe
df = pd.read_csv('data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Splitting the dataset into the Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Creating a linear regression model instance
model = LinearRegression()

# Fitting the data into the model/training the model
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Saving model using pickle
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))
print(model.predict([[20]]))
