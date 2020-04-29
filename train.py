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
patients_directory = "~\\COVID-19 Radiography Database\\patients\\"


def rank(filename):
    radio = mpimg.imread(filename)
    plt.figure()
    plt.imshow(radio, cmap='gray')
    plt.show()
    X_train = radio.ravel()
    print("length ==" + str(len(X_train)))
    pred = model.predict([X_train])
    print("predictions == " + str(pred))


def train(filename, coef):
    radio = mpimg.imread(filename)
    X_train = radio.ravel()
    model.fit([X_train], [coef])


def trainRadios(directory, coef):
    radios = []
    for filename in os.listdir(directory):
        train(directory + filename, coef)
        continue


def rankPatients(directory):
    for filename in os.listdir(directory):
        rank(directory + filename)
        continue


print("training stage")
trainRadios(covid_directory, 1)
print("end training stage")

print("training stage")
trainRadios(normal_directory, 0)
print("end training stage")


print("predicting stage")
rankPatients(patients_directory)
print("end predicting stage")
