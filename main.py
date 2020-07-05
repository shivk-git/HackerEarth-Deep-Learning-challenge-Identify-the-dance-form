# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import cv2
import os, sys
from tqdm import tqdm

train = pd.read_csv("/content/drive/My Drive/Datasets/Dance/train.csv")
test = pd.read_csv("/content/drive/My Drive/Datasets/Dance/test.csv")

train.head()

IMG_SIZE = 224
def read_image(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img = cv2.resize(img,(IMG_SIZE, IMG_SIZE),interpolation = cv2.INTER_AREA)
    return img

train_path = '/content/drive/My Drive/Datasets/Dance/train/'
test_path = '/content/drive/My Drive/Datasets/Dance/test/'

train_img = []
for Image in tqdm(train['Image'].values):
    train_img.append(read_image(train_path + Image))

# namolization of images
x_train = np.array(train_img, np.float64)/255.
x_train.shape

# target variable
label_list = train['target'].tolist()
label_numeric = {k: v for v, k in enumerate(set(label_list))}
y_train = [label_numeric[k] for k in label_list]
y_train = np.array(y_train)

for v, k in enumerate(set(label_list)):
  print(v, " ", k)

print(label_numeric)

print(y_train.shape)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
input_shape = (IMG_SIZE, IMG_SIZE, 3)
print(input_shape)

import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau


## CNN without using transfer learning
model = Sequential()
model.add(Conv2D(32, (5,5),padding = 'Same', activation ='relu', input_shape = (224,224,3)))
model.add(Conv2D(32, (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(64, (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(8, activation = "softmax"))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
model.summary()


## CNN using Transfer learning with VGG16 
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(8, activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
model.summary()

x_train.shape[0]

epochs = 1    #epochs = 5-10 for 90+% accuracy
batch_size = 32

train_datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.1, width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,)
train_datagen.fit(x_train)


history = model.fit_generator(
      train_datagen.flow(x_train, y_train, batch_size=batch_size),
      steps_per_epoch = x_train.shape[0],
      epochs=epochs
)

x_train.shape

test_img = []
for Image in tqdm(test['Image'].values):
    test_img.append(read_image(test_path + Image))

x_test = np.array(test_img, np.float64)/255.
x_test.shape

## predict test data
predictions = model.predict(x_test)

# get labels
predictions = np.argmax(predictions, axis=1)
rev_y = {v:k for k,v in label_numeric.items()}
pred_labels = [rev_y[k] for k in predictions]

print(pred_labels)


## save csv file
sub = pd.DataFrame()
sub['Image'] = test['Image']
sub['target'] = pred_labels
filename = 'solution1.csv'
sub.to_csv(filename, mode='w',index=False) 
sub.head()

