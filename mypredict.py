import os
import numpy as np
#import tensorflow
#from keras.preprocessing.image import  img_to_array
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
import cv2


# model=tensorflow.keras.models.load_model('mymodel.h5')

from keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')
print("model is loaded\n")


def getpredictions(filename):
    imgpath='Uploaded_temp/'+filename
    img=cv2.imread(imgpath)
    img=cv2.resize(img,(224,224))

    img=image.img_to_array(img)
    #np.resize(img,(224,224))
    img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
    img=preprocess_input(img,mode='caffe')

    predictions=model.predict(img)
    prediction_result=decode_predictions(predictions,top=1)
    return prediction_result[0][0]
