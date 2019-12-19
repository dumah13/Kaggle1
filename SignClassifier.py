#importing modules
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.patches as patches
import random
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#setting number of train data to show
head_num = 8

#initializing randomness (sounds cool)
random.seed()

#importing training data from csv files
train_data = pd.read_csv("Train.csv")
test_data = pd.read_csv('Test.csv')
train_data_len = len(train_data)
test_data_len = len(test_data)

#printing length of dataset
print('Length of training data frame: ' + str(train_data_len))

#separating labels from training data
train_labels = train_data.pop('ClassId')
image_paths = train_data.pop('Path')
test_paths = pd.Series(['Test/' + x for x in test_data.Path])

#setting directories paths
train_dir = 'Train'
test_dir = 'Test' #Test dit has to be a subdirecoty of Test (Test/Test) because of how ImageDataGenerator works

#Functions to plot images with their label image
def add_image(ax, index,data,paths,labels):
    img = mpimg.imread(paths[index])
    rect = patches.Rectangle(xy=(data.iloc[index, 2],data.iloc[index, 3]),
                             width=data.iloc[index, 4] - data.iloc[index, 2],
                             height= data.iloc[index, 5] - data.iloc[index, 3],
                             linewidth=1,edgecolor='r',facecolor='none')
    ax.set_xlabel('I: ' + str(index ) + ' C: ' + str(labels[index]))
    ax.add_patch(rect)
    ax.imshow(img)

def add_label(ax, label):
    img = mpimg.imread('Meta/' + str(label) + '.png')
    ax.set_xlabel('<--- ' + str(label))
    ax.imshow(img)

#plot rectangle setup
rect_width = 4
rect_height = head_num*2//rect_width

fig,ax = plt.subplots(rect_width, rect_height)

#showing training data
for i,j in ((x,y) for x in range(0,rect_height) for y in range(0, rect_width,2)):
    index = random.randint(0, train_data_len)
    add_image(ax[i][j], index, train_data, image_paths, train_labels)
    add_label(ax[i][j+1], train_labels[index])
      
plt.tight_layout()

plt.show()

#setting model parameters
batch_size = 100
epochs = 10

#setting force image size
IMG_HEIGHT = 30
IMG_WIDTH = 30

#preparing generators for training and test data
train_image_generator = ImageDataGenerator(rescale=1./255, validation_split = 0.2,) # Generator for our training data
test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(
                                                           batch_size=batch_size,
                                                           directory=train_dir,
                                                           color_mode = "rgba",
                                                           shuffle=True,
                                                           classes = [str(x) for x in range(0, 43)],
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           subset = 'training')

val_data_gen =  train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           color_mode = "rgba",
                                                           classes = [str(x) for x in range(0, 43)],
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           subset = "validation")

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=test_dir,
                                                           color_mode = "rgba",
                                                           shuffle = False,
                                                           class_mode = None,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH)) 
    

#model (i kinda guessed the layers and model hyperparameters, then validated it with what other people had done
#the i readjusted them and ended up with this model)
model = Sequential([
    Conv2D(16, 4, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,4)),
    Dropout(0.25),
    Conv2D(32, 4, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(43, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

#training the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(train_data_len*0.8) // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(train_data_len*0.2) // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

#plotting a nice Accuracy(Epoch) plot (stolen from a toturial)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions = model.predict_generator(test_data_gen,
                                     verbose = 1)
predictions_args = pd.Series([np.argmax(x) for x in predictions])
 
prediction_dict = {'ClassId' : predictions_args, 'Path' : test_data.Path}

predictions_labeled = pd.DataFrame(prediction_dict)
predictions_labeled.to_csv("predictionsResults.csv", index = False)

fig,ax = plt.subplots(rect_width, rect_height)

#showing the results with labels
for i,j in ((x,y) for x in range(0,rect_height) for y in range(0, rect_width,2)):
    index = random.randrange(0, test_data_len)
    add_image(ax[i][j], index, test_data, test_paths,  predictions_args)
    add_label(ax[i][j+1], predictions_labeled.iloc[index, 0])
      
plt.tight_layout()

plt.show()

