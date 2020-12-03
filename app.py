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
from matplotlib.pyplot import hist

import plotly.express as px
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
import streamlit.components.v1 as components

from skimage.io import imread
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.measure import block_reduce
from skimage.color import rgb2gray
import imageio
from skimage import color, io
import scipy.ndimage as nd





def main():
    # Register pages
    pages = {
        "Home": Home,
        "Graphs": Graph,
        "Labels": Predict_by_label,
        "Prediction_model": Prediction_model,

    }
    st.sidebar.title("Image Analysis")
    page = st.sidebar.selectbox("Select Menu", tuple(pages.keys()))
    pages[page]()

def Home():

    def main():
        st.markdown("<h1 style='text-align: center; color: #002966;'>Image Classification</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #002966;'>Identifying Patients with Pneumonia</h1>", unsafe_allow_html=True)
        st.write(
        """
        Deep Learning - Learn Neural Network Modeling! 
        
        Use Image classification to identify healty or not lung (Pneumonia) use Neural Network models:
        -  Image recognintion uses Artificial Intelligent technology to identify objects.
        -  Requieres Big data as image labels and a lot of computing power.
        -  TensorFlow create a convolutional layer and final prediction.
        -  CNN is a architecture designed to process and correlate images. 
        -  Image preparation include: 
                    - Image size.
                    - Number of images.
                    - Aspect ratio.
                    - Image scaling. 
                    - Mean.
                    - Standard deviation of input date.
                    - Normalizing image inputs. 
                    - Dimensionality reduction.
        ---
        """)
        st.markdown("<h1 style='text-align: center; color: #002966;'>Process Convolutional: Recognition by Artificial Intelligence, applied to analyzing visual image.</h1>", unsafe_allow_html=True)
        img = Image.open('convolutional.png')
        st.image(img, width=700)
        page_bg_img = '''
            <style>
            body {
            background-image: url("https://images.pexels.com/photos/1103970/pexels-photo-1103970.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1000");
            background-size: cover;
            }
            </style>
            '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        footer_temp1 = """
            <!-- CSS  -->
            <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
            <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
            <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
            <footer class="background-color: gray">
                <div class="container" id="About App">
                <div class="row">
                    <div class="col l6 s12">
                    </div>
            <div class="col l3 s12">
                    <h5 class="black-text">Connect With Me</h5>
                    <ul>
                        <a href="http://www.monicadatascience.com/" target="#002966" class="black-text">
                        ❤<i class="❤"></i>
                    </a>
                    <a href="https://www.linkedin.com/in/monica-bustamante-b2ba3781/3" target="#002966" class="black-text">
                        <i class="fab fa-linkedin fa-4x"></i>
                    <a href="http://www.monicadatascience.com/" target="#002966" class="black-text">
                    ❤<i class="❤"></i>
                    </a>
                    <a href="https://github.com/Moly-malibu/Image Analysis" target="#002966" class="black-text">
                        <i class="fab fa-github-square fa-4x"></i>
                    <a href="http://www.monicadatascience.com/" target="#002966" class="black-text">
                    ❤<i class="❤"></i>
                    </a>
                    </ul>
                    </div>
                </div>
                </div>
                <div class="footer-copyright">
                <div class="container">
                Made by <a class="black-text text-lighten-3" href="http://www.monicadatascience.com/">Monica Bustamante</a><br/>
                <a class="black-text text-lighten-3" href=""> @Copyrigh</a>
                </div>
                </div>
            </footer>
            """
        components.html(footer_temp1,height=500)

    if __name__ == "__main__":
        main()

title_temp = """
	 <!-- CSS  -->
	  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
	  <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	  <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	   <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
	 <footer class="background-color: gray">
	    <div class="container" id="About App">
	      <div class="row">
	        <div class="col l6 s12">
                <h4 class="black-text">Machine Learning and Bid Data</h5>
	          <p class="grey-text text-lighten-4">Using Streamlit, Pandas, Numpy, Seaborn, Tensorflow,  Keras, Matplotlib.</p>
	        </div>     
	  </footer>
	"""
components.html(title_temp,height=100)
    

def Graph():
    page_bg_img = '''
            <style>
            body {
            background-image: url("https://images.pexels.com/photos/288100/pexels-photo-288100.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1000");
            background-size: cover;
            }
            </style>
            '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
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
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.write('***Statistical Data Analysis***')
    st.table(train_data.describe())

def Predict_by_label():

    page_bg_img = '''
            <style>
            body {
            background-image: url("https://images.pexels.com/photos/288100/pexels-photo-288100.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1000");
            background-size: cover;
            }
            </style>
            '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    # #load data
    master = os.listdir('chest_xray/chest_xray')
    train = 'chest_xray/chest_xray/train/'
    val= 'chest_xray/chest_xray/val/'
    test= 'chest_xray/chest_xray/test/'

    #Manual selection sampling
    img_name = 'NORMAL2-IM-0407-0001.jpeg'
    img_normal = load_img('chest_xray/chest_xray/train/NORMAL/' + img_name)
    st.write('NORMAL')
    plt.imshow(img_normal)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


    img_name = 'person111_virus_212.jpeg'
    img_pneumonia = load_img('chest_xray/chest_xray/train/PNEUMONIA/' + img_name)
    st.write('PNEUMONIA')
    plt.imshow(img_pneumonia)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


    #Train
    os.listdir(train)
    train_normal=train+'NORMAL/'
    train_pneumonia=train+'PNEUMONIA/'

    num_healthy_tr = len(os.listdir(train))
    num_pneumonia_tr = len(os.listdir(train))

    num_healthy_val = len(os.listdir(val))
    num_pneumonia_val = len(os.listdir(val))

    total_train= num_healthy_tr +  num_pneumonia_tr
    total_val = num_healthy_val + num_pneumonia_val

    train_sample = 5217
    val_sample = 17
    epochs = 20
    batch_size = 16

    hale_rand = np.random.randint(0, len(os.listdir(train_normal)))
    hale_pic = os.listdir(train_normal)[hale_rand]
    st.write('***Healthy lung***:', hale_pic)

    hale_test = train_normal + hale_pic

    #Lung with Pneumonia
    Pneumonia_rand = np.random.randint(0, len(os.listdir(train_pneumonia)))
    sick = os.listdir(train_pneumonia)[hale_rand]
    sick_test = train_pneumonia + sick
    st.write('***Positive for Pneumonia***:', sick)

    healthy_lung = Image.open(hale_test)
    sick_lung = Image.open(sick_test)

    #Sampling Lung, random selection
    lung = plt.figure(figsize=(11,4))
    test_a1 = lung.add_subplot(1,2,1)
    img_lung = plt.imshow(healthy_lung)
    test_a1.set_title('Healthy Lung')
    test_a2 = lung.add_subplot(1,2,2)
    img_lung= plt.imshow(sick_lung)
    test_a2.set_title('Positive for Pneumonia') 
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()



def Prediction_model():
    page_bg_img = '''
            <style>
            body {
            background-image: url("https://images.pexels.com/photos/288100/pexels-photo-288100.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1000");
            background-size: cover;
            }
            </style>
            '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    data_set = Path('chest_xray/chest_xray')
    train_1 = data_set / 'train'
    val_1 = data_set / 'val'
    test_1 = data_set / 'test'
    normal_lung = train_1 / 'NORMAL'
    pneumonia_lung = train_1 / 'PNEUMONIA'

    #Model
    batch_size = 16
    epochs = 50
    img_height = 224
    img_width = 224
    train_sample = 5217
    val_sample = 17

    train_data = ImageDataGenerator(rescale=1./255)
    test_data = ImageDataGenerator(rescale=1./255)

    #Model
    resnet = ResNet50(weights='imagenet', include_top=False )

    for layer in resnet.layers:
        layer.trainable = False

    x = resnet.output
    x = GlobalAveragePooling2D()(x) #layer flatten
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(resnet.input, predictions)

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    resnet.summary()

# Setup Architecture
    custom_cnn = Sequential([
        Conv2D(64, (3,3), input_shape=(224, 224, 3), padding='same', activation='relu'),
        MaxPooling2D((2,2), padding='same'),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(16, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(8,(2,2), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    custom_cnn.summary()

    train_data = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_data = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_data.flow_from_directory(
        train_1,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_data.flow_from_directory(
        val_1,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    test_generator = test_data.flow_from_directory(
    test_1,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

    history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_sample // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_sample// batch_size)

    # #Save Model
    # model.save('saved_model/img_rx')

    # evaluate the model
    scores = model.evaluate_generator(test_generator)
    st.write("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    sns.lineplot(data=pd.DataFrame(history.history));

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    predictions = model.predict(test_generator)
    np.argmax(predictions[0])


    lung = train_generator[0][0][1]
    plt.imshow(lung);
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    pneumonia = train_generator[0][0][0]
    plt.imshow(pneumonia);
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    rx_grayscale = rescale_intensity(color.rgb2gray(lung))
    rx_grayscale.shape

    plt.imshow(rx_grayscale, cmap="gray");
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    rx_grayscale.shape
    horizontal_edge_convolution = np.array([[1,1,1],
                                        [0,0,0],
                                        [-1,-1,-1]])

    vertical_edge_convolution = np.array([[1, 0, -1],
                                        [1, 0, -1],
                                        [1, 0, -1]])
    rx_horizontal = nd.convolve(rx_grayscale, horizontal_edge_convolution)
    rx_vertical = nd.convolve(rx_grayscale, vertical_edge_convolution)
    plt.figure(figsize=(30,10))

    labels = ["Orginal", "Horizontal Edges", "Vertical Edges"]
    images = [rx_grayscale, rx_horizontal, rx_vertical]
    i = 0
    for label, image in zip(labels, images):

        plt.subplot(1,3,i+1)
        plt.grid(False)
        plt.imshow(image, cmap="gray")
        plt.title(label)
        i += 1 
    plt.show()
    reduced = block_reduce(rx_grayscale, (2,2), np.max)
    plt.imshow(reduced, cmap="gray");
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

if __name__=='__main__':
    main()


#https://git-lfs.github.com/
