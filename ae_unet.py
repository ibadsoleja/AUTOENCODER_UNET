from matplotlib import pyplot as plt
import numpy as np
import cv2 
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import img_to_array

import os
from keras.models import Model
from matplotlib import pyplot as plt

SIZE = 256
from tqdm import tqdm 

img_data = []
path1 = 'Pylon_imgaes\\DIN_Gray'
files = os.listdir(path1)
for i in tqdm(files):
    img = cv2.imread(path1+'\\'+i, 1)
    img = cv2.resize(img,(SIZE, SIZE))
    img_data.append(img_to_array(img))
img_array = np.reshape(img_data, (len(img_data),SIZE, SIZE, 3))
img_array = img_array.astype('float32')/255.

from models import build_autoencoder, build_decoder, build_encoder, build_unet
autoencoder_model = build_autoencoder(img.shape)
autoencoder_model.compile(optimizer = 'adam', loss= 'mean_squared_error', metrics= ['accuracy'])
print(autoencoder_model.summary())

history = autoencoder_model.fit(img_array, img_array, epochs= 1000, verbose = 1, batch_size= 32)

autoencoder_model.save('autoencoder_pylon_grayscale_img_5epochs.h5')

