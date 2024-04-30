from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# resnet
from keras import applications
from keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dropout,GlobalAveragePooling2D
from keras.models import Model

# define resnet model
def define_resnet(num_classes,RESOLUTION,learningrate=.0001):
    base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (RESOLUTION,RESOLUTION,1))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    adam = Adam(learning_rate=learningrate)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def define_base(classnums,RESOLUTION,learningrate=.0001):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', input_shape=(RESOLUTION, RESOLUTION, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(Dense(classnums, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=learningrate, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
