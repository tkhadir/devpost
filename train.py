# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:44:33 2020

@author: TKhadir
"""
import scipy as sc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.linear_model import LinearRegression
import numpy as np
from skimage.transform import resize
from skimage import data, color

model = LinearRegression()

covid_directory = "~\\COVID-19 Radiography Database\\COVID-19\\"
normal_directory = "~\\COVID-19 Radiography Database\\NORMAL\\"
patients_directory = "~\\COVID-19 Radiography Database\\patients\\"


def rank(filename):
    radio = color.rgb2gray(mpimg.imread(filename))
    plt.figure()
    plt.imshow(radio, cmap='gray')
    plt.show()
    X_train = [radio.ravel()]
    print("predictions == " + str(model.predict(X_train)))


def train(filename, coef):
    radio = color.rgb2gray(mpimg.imread(filename))
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
