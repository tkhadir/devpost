import scipy as sc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression()


covid_directory = "~\\COVID-19 Radiography Database\\COVID-19\\"
normal_directory = "~\\COVID-19 Radiography Database\\NORMAL\\"
patients_directory= "~\\COVID-19 Radiography Database\\patients\\"

def train(x, y):
    model.fit(x, y)

def score(x, y):
    return model.score(x, y)

def predict(x):
    model.predict(x)

def rank(filename):
    radio = mpimg.imread(filename)
    aradio = np.array(radio)
    plt.figure()
    plt.imshow(radio, cmap='gray')
    plt.show()
    X=aradio.ravel()
    length = len(X)
    X_train = X.reshape(length, 1)
    predictions = predict(X_train)
    print("predictions == " + str(predictions))

def analyze(filename, plt):
    radio = np.array(mpimg.imread(filename))
    X=radio.ravel()
    length = len(X)
    y=np.full((length), 1)
    X_train = X.reshape(length, 1)
    train(X_train, y)
    #print("score == " + str(score(X_train, y)))
    plt.scatter(X, y)
    

def getRadios(directory):
    plt.figure()
    for filename in os.listdir(directory):
        analyze(directory + filename, plt)
        continue
    plt.show()

def rankPatients(directory):
    for filename in os.listdir(directory):
        rank(directory + filename)
        continue


print("training stage")        
getRadios(covid_directory)
print("end training stage")
print("training stage")        
getRadios(normal_directory)
print("end training stage")

print("predicting stage")
rankPatients(patients_directory)
print("end predicting stage")
