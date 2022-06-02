import numpy as np
from PIL import Image, ImageOps
import scipy.io
import pandas as pd
import cv2 as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import easygui
import requests
import matplotlib.image as mpimg
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from glob import glob
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from statistics import mean

path_model = "C:/Users/jadel/Documents/OpenClassrooms/Projet 6/vgg_model.hdf5"
vgg_model = keras.models.load_model(path_model)


indices_to_class = {0: 'Afghan_hound',
 1: 'American_Staffordshire_terrier',
 2: 'Chihuahua',
 3: 'Italian_greyhound',
 4: 'Kerry_blue_terrier',
 5: 'Labrador_retriever',
 6: 'Siberian_husky',
 7: 'basset',
 8: 'borzoi',
 9: 'pug'}

def resize2SquareKeepingAspectRation(img, size, interpolation):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv.resize(mask, (size, size), interpolation)

def preprocess_img(w, h):
    img = False
    while img == False:
        #path_input = input("Veuillez entrer l'adresse d'une image de chien: ")

        path_input = easygui.fileopenbox()
        try:
            img = mpimg.imread(path_input) #lire l'image en array
            print("Recherche de fichier local réussie")
            break
        except :
             print("Pas de fichier correspondant en local")
             try:
                 img = np.array(Image.open(requests.get(path_input, stream=True).raw))
                 print("Recherche de fichier sur internet réussie")
                 break
             except :
                print("Pas de fichier correspondant sur internet non plus")

    img = resize2SquareKeepingAspectRation(img, 255, cv.INTER_LINEAR) # redimensionner l'image
    #equilization
    img_yuv = cv.cvtColor(img,cv.COLOR_RGB2YUV) #passage de RGB à YUV
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0]) #egalisation d'histogramme du contraste
    img_equ = cv.cvtColor(img_yuv, cv.COLOR_YUV2RGB)  #passage à nouveau au format RGB
    #img_jpg = Image.fromarray(img_equ)
    return img_equ


def predict_img():

  processed_img = preprocess_img(255,255).reshape(1,255,255,3)
  result = vgg_model.predict(processed_img).argmax()
  return print("La race du chien est :", indices_to_class[result])

predict_img()