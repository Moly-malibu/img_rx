import numpy as np
import pandas as pd
import os
from PIL import Image
from pandas.io.formats import style
import tensorflow as tf
import tensorflow
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import MaxPooling2D
from pathlib import Path
import seaborn as sns
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import streamlit as st



# Load data with path
data_set = Path('chest_xray/chest_xray')
train_1 = data_set / 'train'
val_1 = data_set / 'val'
test_1 = data_set / 'test'
normal_lung = train_1 / 'NORMAL'
pneumonia_lung = train_1 / 'PNEUMONIA'

# Get all the images
normal_0 = normal_lung.glob('*.jpeg')
pneumonia_1 = pneumonia_lung.glob('*.jpeg')

train_data = [] #insert the data into this list

for img in normal_0: #normal cases = 0. 
    train_data.append((img,0))
for img in pneumonia_1: # pneumonia cases = 1. 
    train_data.append((img, 1))
train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None) #convert with pandas dataframe from the data list
train_data = train_data.sample(frac=1.).reset_index(drop=True) # sample and reset index data 
(train_data.info())

# #Plot 
cases_count = train_data['label'].value_counts()
plt.figure(figsize=(10,8))
ax = sns.barplot(x=cases_count, y= cases_count.values, ci=68)
plt.title("X-Ray records of the lungs")
color_patch = mpatches.Patch(color ='#EDB120', label="Pneumonia")
plt.legend(handles=[color_patch])
plt.show()

# #Model
# batch_size = 16
# epochs = 50
# img_height = 224
# img_width = 224
# train_sample = 5217
# val_sample = 17

# train_data = ImageDataGenerator(rescale=1./255)
# test_data = ImageDataGenerator(rescale=1./255)

# #Model
# resnet = ResNet50(weights='imagenet', include_top=False )

# for layer in resnet.layers:
#     layer.trainable = False

# x = resnet.output
# x = GlobalAveragePooling2D()(x) #layer flatten
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(1, activation='sigmoid')(x)
# model = Model(resnet.input, predictions)

# # Setup Architecture

# custom_cnn = Sequential([
#     Conv2D(64, (3,3), input_shape=(224, 224, 3), padding='same', activation='relu'),
#     MaxPooling2D((2,2), padding='same'),
#     Conv2D(32, (3,3), activation='relu'),
#     MaxPooling2D((2,2)),
#     Conv2D(16, (3,3), activation='relu'),
#     MaxPooling2D((2,2)),
#     Conv2D(8,(2,2), activation='relu'),
#     MaxPooling2D((2,2)),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])
# custom_cnn.summary()

# #compile Model
# model.compile(optimizer='rmsprop', 
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])

# train_data = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# train_generator = train_data.flow_from_directory(
#     train_1,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary')

# validation_generator = test_data.flow_from_directory(
#     val_1,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary')

# test_generator = test_data.flow_from_directory(
#     test_1,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary')

# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_sample // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=val_sample// batch_size)

# # evaluate the model
# scores = Model.evaluate(test_generator)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# sns.lineplot(data=pd.DataFrame(history.history));

# #Analysis the Accuracy
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(test_generator, verbose=2)



# lung = train_generator[0][0][1]
# plt.imshow(lung);

# pneumonia = train_generator[0][0][0]
# plt.imshow(pneumonia);

# import imageio
# from skimage.exposure import rescale_intensity
# from skimage import color, io

# rx_grayscale = rescale_intensity(color.rgb2gray(lung))
# rx_grayscale.shape
# plt.imshow(rx_grayscale, cmap="gray");
# rx_grayscale.shape

# horizontal_edge_convolution = np.array([[1,1,1],
#                                         [0,0,0],
#                                         [-1,-1,-1]])

# vertical_edge_convolution = np.array([[1, 0, -1],
#                                      [1, 0, -1],
#                                      [1, 0, -1]])

# import scipy.ndimage as nd
# rx_horizontal = nd.convolve(rx_grayscale, horizontal_edge_convolution)
# rx_vertical = nd.convolve(rx_grayscale, vertical_edge_convolution)

# rx_horizontal.shape
# rx_grayscale[0,0]

# plt.figure(figsize=(30,10))

# labels = ["Orginal", "Horizontal Edges", "Vertical Edges"]
# images = [rx_grayscale, rx_horizontal, rx_vertical]

# i = 0
# for label, image in zip(labels, images):

#     plt.subplot(1,3,i+1)
#     plt.grid(False)
#     plt.imshow(image, cmap="gray")
#     plt.title(label)
#     i += 1 

# plt.show()

# #load data
# master = os.listdir('chest_xray/chest_xray')
# train = 'chest_xray/chest_xray/train/'
# val= 'chest_xray/chest_xray/val/'
# test= 'chest_xray/chest_xray/test/'

# #Manual selection sampling
# img_name = 'NORMAL2-IM-0407-0001.jpeg'
# img_normal = load_img('chest_xray/chest_xray/train/NORMAL/' + img_name)
# print('NORMAL')
# plt.imshow(img_normal)
# plt.show()

# img_name = 'person111_virus_212.jpeg'
# img_pneumonia = load_img('chest_xray/chest_xray/train/PNEUMONIA/' + img_name)
# print('PNEUMONIA')
# plt.imshow(img_pneumonia)
# plt.show()


# #Train
# os.listdir(train)
# train_normal=train+'NORMAL/'
# train_pneumonia=train+'PNEUMONIA/'

# num_healthy_tr = len(os.listdir(train))
# num_pneumonia_tr = len(os.listdir(train))

# num_healthy_val = len(os.listdir(val))
# num_pneumonia_val = len(os.listdir(val))

# total_train= num_healthy_tr +  num_pneumonia_tr
# total_val = num_healthy_val + num_pneumonia_val

# train_sample = 5217
# val_sample = 17
# epochs = 20
# batch_size = 16

# hale_rand = np.random.randint(0, len(os.listdir(train_normal)))
# hale_pic = os.listdir(train_normal)[hale_rand]
# print('Healthy lung :', hale_pic)

# hale_test = train_normal + hale_pic

# #Lung with Pneumonia
# Pneumonia_rand = np.random.randint(0, len(os.listdir(train_pneumonia)))
# sick = os.listdir(train_pneumonia)[hale_rand]
# sick_test = train_pneumonia + sick
# print('Positive for Pneumonia:', sick)

# healthy_lung = Image.open(hale_test)
# sick_lung = Image.open(sick_test)

# #Sampling Lung, random selection
# lung = plt.figure(figsize=(11,4))
# test_a1 = lung.add_subplot(1,2,1)
# img_lung = plt.imshow(healthy_lung)
# test_a1.set_title('Healthy Lung')
# test_a2 = lung.add_subplot(1,2,2)
# img_lung= plt.imshow(sick_lung)
# test_a2.set_title('Positive for Pneumonia') 
# plt.show()

# data = [['Normal', len(train_normal)], ['Pneumonia', len(train_pneumonia)]]
# df = pd.DataFrame(data, columns=['Class', 'Count'])

# sns.barplot(x=df['Class'], y=df['Count']);