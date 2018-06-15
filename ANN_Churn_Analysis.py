#ANN
# Part-1 Data preprocessing
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1]  = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ohe = OneHotEncoder(categorical_features = [1])
X = ohe.fit_transform(X).toarray()
X= X[:, 1:]

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part-2 Building ANN

#importing keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising ANN
classifier = Sequential()

#adding the input and first hidden layer
classifier.add(Dense(6,input_shape=(11,), kernel_initializer = 'uniform', activation='relu' ))

#adding 2nd hidden layer
classifier.add(Dense(6, kernel_initializer = 'uniform', activation='relu' ))

#adding output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation='sigmoid' ))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs=100)

#Part-3 Prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

