#implementation of CNN BiLSTM Model for violence detection:

import os
import cv2
import time
import numpy as np
from tqdm import tqdm
from random import shuffle
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed, Reshape
from tensorflow.keras.layers import  Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

#Directories of the raw dataset (videos corresponding to the data)
hviolence = 'HockeyFightsvideos/violence/'
hnoviolence = 'HockeyFightsvideos/noviolence/'

frames = 20 #NUmber of frames given in single pass to the network
img_shape = 200

#---------Network---------------
#CNN
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(20, img_shape, img_shape, 3), padding="same"))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling3D((1, 2,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling3D((1, 2,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling3D((1, 2,2)))
model.add(Reshape((20, 10816))) 
#BiLSTM
lstmF = LSTM(units=32)
lstmB = LSTM(units=32, go_backwards = True)
model.add(Bidirectional(lstmF, backward_layer = lstmB))
#Dense 
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
#-------------------------------
#this function will join the extracted frames into the data frame directory
def classifytodfs(violence, noviolence, path):
    count = 0
    for files in os.listdir(violence):
        cap = cv2.VideoCapture(os.path.join(violence, files))
        sucess, image = cap.read()
        sucess = True
        while sucess:
            sucess, image = cap.read()
            if not sucess:
                break 
            cv2.imwrite(path+str(count)+".jpg",image)
            if cv2.waitKey(10) == 27:
                break
            count += 1
    for files in os.listdir(noviolence):
        cap = cv2.VideoCapture(os.path.join(noviolence, files))
        sucess, image = cap.read()
        sucess = True
        while sucess:
            sucess, image = cap.read()
            if not sucess:
                break 
            cv2.imwrite(path+str(count)+".jpg",image)
            if cv2.waitKey(10) == 27:
                break
            count += 1

classifytodfs(hviolence, hnoviolence, path = "Data/Hfights/DF/")

def dataframe():
    dataset = []
    image = []
    limit = 0
    c = 0

    for file in tqdm(os.listdir('Data/Hfights/DF/')):
        path = os.path.join('Data/Hfights/DF/', file)
        img = cv2.resize(cv2.imread(path), (200, 200))
        image.append(np.array(img))
        limit += 1
        c += 1
        if c == frames:
            c = 0
            if limit < 20056:
                dataset.append([image, np.array([1, 0])])
            elif limit >= 20056:
                dataset.append([image, np.array([0, 1])])
            image = []
    
    shuffle(dataset)
    np.save('dataset.npy', dataset)
    print(dataset)
    return dataset


df = dataframe()

data = np.load('dataset.npy', allow_pickle=True)
train, test = train_test_split(data, train_size = 0.9)
X = np.array([i[0] for i in train]).reshape(-1, 20, img_shape, img_shape, 3)
Y = np.array([i[1] for i in train])
x_valid = np.array([i[0] for i in test]).reshape(-1, 20, img_shape, img_shape, 3)
y_valid = np.array([i[1] for i in test])
X = X.astype('float32')/255
x_valid = x_valid.astype('float32')/255


model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics = ['accuracy'])
Tensorboard = TensorBoard(log_dir='logs/{}'.format('Model {}'.format(int(time.time()))))
model.fit( X,Y, epochs=15, validation_data=(x_valid, y_valid),batch_size=16, verbose=1, callbacks=[Tensorboard])
model.save('CNN-BiLSTM.h5', overwrite=True, include_optimizer=True)

#Testing:
model = load_model(‘CNN-BiLSTM200.h5’)
data = dataframe(folder=’rawdata/dataframes’)
data = np.load(‘dataset.npy’,allow_pickle=True)
X = np.array(i[0] for i in data).reshape(-1, 20, img_shape, image_shape,3)
Y = np.array(i[1] for i in data)
X = X.astype(‘float32’)/255
model.evaluate(np.array(X),np.array(Y), batch_size=16, verbose=1)

