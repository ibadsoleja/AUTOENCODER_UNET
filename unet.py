from models import build_autoencoder, build_encoder, build_unet
from keras.models import load_model 
import numpy as np
import cv2
#from keras.preprocessing.image import img_to_array
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Input, Conv2DTranspose
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
import os
from keras.models import Model
from matplotlib import pyplot as plt

autoencoder_model = load_model('autoencoder_pylon_grayscale_img_5epochs.h5', compile = False)
# Define Encoder without the Decoder
input_shape = (256, 256, 3)
input_img = Input(shape = input_shape)

encoder = build_encoder(input_img)
encoder_model = Model(input_img, encoder)
print (encoder_model.summary())

num_encoder_layers = len(encoder_model.layers)

###IMPLEMENT UNET###
input_shape = (256, 256, 3)
unet_model = build_unet(input_shape)
unet_layer_names = []
for layer in unet_model.layers:
    unet_layer_names.append(layer.name)
autoencoder_layer_names = []
for layer in autoencoder_model.layers:
    autoencoder_layer_names.append(layer.name)

###Transferring weights to UNET now
for l1, l2 in zip(unet_model.layers[:35], autoencoder_model.layers[0:35]):
    l1.set_weights(l2.get_weights())
from keras.optimizers import Adam
import segmentation_models as sm
#sm.set_framework('tensorflow.keras')
#unet_model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss, metrics = [sm.metrics.iou_score])#FOR MULTI CLASS
unet_model.compile('Adam', loss=sm.losses.binary_focal_jaccard_loss, metrics = [sm.metrics.iou_score])#FOR BINARY 
unet_model.summary()
print(unet_model.output_shape)
unet_model.save('modified_ae_unet_model_weights.h5')

