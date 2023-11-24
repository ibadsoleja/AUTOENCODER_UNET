import cv2
import numpy as np
from matplotlib import pyplot as plt
#########################################################################
#Load data for U-net training. 
#################################################################
import os
image_directory = 'Pylon_imgaes\\DIN_Gray'
mask_directory = 'Pylon_imgaes\\DIN'


SIZE = 256
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.
images = os.listdir(image_directory)
masks = os.listdir(mask_directory)

for image_name in images:
    if image_name.endswith('.jpg'):
        corresponding_mask_name = image_name.replace('.jpg', '_mask.png')
        if corresponding_mask_name in masks:
            img_path = os.path.join(image_directory, image_name)
            mask_path = os.path.join(mask_directory, corresponding_mask_name)

            image = cv2.imread(img_path, 1)
            mask = cv2.imread(mask_path, 0)

            if image is not None and mask is not None:
                # Resize both image and mask to the desired SIZE
                image_resized = cv2.resize(image, (SIZE, SIZE))
                mask_resized = cv2.resize(mask, (SIZE, SIZE))

                image_dataset.append(image_resized)
                mask_dataset.append(mask_resized)
            else:
                print(f"Error reading {image_name} or its mask.")

# Normalize images
image_dataset = np.array(image_dataset) / 255.
# Adjusting the dimension expansion:
mask_dataset = np.expand_dims(np.array(mask_dataset), axis=3) / 255.

print("Image filenames:", images[:5])
print("Mask filenames:", masks[:5])

print(len(image_dataset))
print(len(mask_dataset))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.20, random_state = 0)

#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256, 3)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()
########################################################################
#######################################################################
#Load unet model and load pretrained weights
from models import build_autoencoder, build_encoder, build_unet
from keras.optimizers import Adam
import segmentation_models as sm

input_shape = (256, 256, 3)
pre_trained_unet_model = build_unet(input_shape)
pre_trained_unet_model.load_weights('modified_ae_unet_model_weights.h5')
pre_trained_unet_model_weights = pre_trained_unet_model.get_weights()[0][1]


# random_wt_unet_model.compile('Adam', loss=sm.losses.binary_focal_jaccard_loss, metrics=[sm.metrics.iou_score])
pre_trained_unet_model.compile('Adam', loss=sm.losses.binary_focal_jaccard_loss, metrics=[sm.metrics.iou_score])

####################################################################

#train both models

#Train the model
batch_size=2


pre_trained_unet_model_history = pre_trained_unet_model.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=5)

pre_trained_unet_model.save('pre_trained_unet_model_5epochs.h5')

#PLot history to see which one converged fast
history = pre_trained_unet_model_history

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['iou_score']
val_acc = history.history['val_iou_score']
plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()
################################################################################
#For each model check the IoU and verify few random images
from keras.models import load_model


pre_trained_unet_model = load_model('pre_trained_unet_model_5epochs.h5', compile=False)

# my_model = random_wt_unet_model
my_model = pre_trained_unet_model

import random
test_img_number = random.randint(0, X_test.shape[0]-1)
#test_img_number = 119
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]

test_img_input=np.expand_dims(test_img, 0)
prediction = (my_model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()


#IoU for a single image
from tensorflow.keras.metrics import MeanIoU
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(ground_truth[:,:,0], prediction)
print("Mean IoU =", IOU_keras.result().numpy())


#Calculate IoU for all test images and average
 
import pandas as pd

IoU_values = []
for img in range(0, X_test.shape[0]):
    temp_img = X_test[img]
    ground_truth=y_test[img]
    temp_img_input=np.expand_dims(temp_img, 0)
    prediction = (my_model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    
    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(ground_truth[:,:,0], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)

    print(IoU)
    


df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]    
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU)