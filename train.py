
# python 2/3 compatibility
from __future__ import print_function
import numpy as np
# simplified interface for building models
import keras
# our handwritten character labeled dataset
from keras.datasets import mnist
# because our models are simple
from keras.models import Sequential
# dense means fully connected layers, dropout is a technique to improve convergence, flatten to reshape our matrices for feeding
# into respective layers
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
# for convolution (images) and pooling is a technique to help choose the most relevant features in an image
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from scipy.misc import imread, imresize
import os
from sklearn.preprocessing import LabelEncoder

path = 'ANGLE_DETECTION/identifyRotation/'

data = []
labels = []

img_rows, img_cols = 28, 28

batch_size = 128
# 10 difference characters
num_classes = 4
# very short training time
epochs = 250


for folder, subfolders, files in os.walk(path):
  for name in files:
    if name.endswith('.jpg'):
      x = imread(folder + '/' + name)
      x = imresize(x, (img_rows, img_cols))
      print (x.shape)
      data.append(x)

      labels.append(os.path.basename(folder))

# mini batch gradient descent ftw

# input image dimensions
# 28x28 pixel images.


# the data downloaded, shuffled and split between train and test sets
# if only all datasets were this easy to import and format
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# this assumes our data format
# For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while
#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).


x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                    random_state=0,
                                                    test_size=0.2)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


input_shape = (img_rows, img_cols, 3)
# build our model
model = Sequential()
# convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# again
model.add(Conv2D(64, (3, 3), activation='relu'))
# choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))
# flatten since too many dimensions, we only want a classification output
model.add(Flatten())
# fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
# one more dropout for convergence' sake :)
model.add(Dropout(0.5))
# output a softmax to squash the matrix into output probabilities
model.add(Dense(num_classes, activation='softmax'))
# Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
# categorical ce since we have multiple classes (10)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# train that ish!


lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
# how well did it do?
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Save the model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
  json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


###############################################################################
# PREDICTION
#############################################################################@##

def detect_angle_cnn(filename):
    # global  loaded_model
  print('^^^^^^^^', filename)
  if filename == '/tmp/dummy.jpg':
    return filename
  graph1 = Graph()
  with graph1.as_default():
    session1 = Session()
    with session1.as_default():
      json_file = open(model_json_path, 'r')
      loaded_model_json = json_file.read()
      json_file.close()

      loaded_model = model_from_json(loaded_model_json)
      # load woeights into new model
      loaded_model.load_weights(model_path)
      print("*****Loaded Model from disk******")
      fpth = os.path.splitext(filename)[0] + '_de_skewed.jpg'

      im = cv2.imread(filename)

      x = imresize(im, (28, 28))
      x = x.reshape(1, 28, 28, 3)
      out = loaded_model.predict(x)
      angle = dict_normal[np.argmax(out, axis=1)[0]]
      del loaded_model
      gc.collect()
      print ('Rotation {} Degree'.format(angle))
      img_rotated = ndimage.rotate(im, int(angle))
      cv2.imwrite(fpth, img_rotated)
  print('FPATH: ', fpth)
  print('***Deskew done***')
  return fpth
