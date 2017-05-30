import numpy as np
import os.path
import sys
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
#from keras.layers.convolutional import Conv2D, Conv3D
#from keras.layers.pooling import MaxPooling2D, MaxPooling3D
from keras.layers.core import Flatten, Dense, Lambda
from keras import backend as K
from keras.optimizers import SGD, Adagrad, Adadelta
from keras.callbacks import CSVLogger

airports = ['EIDW', 'EGLL', 'LFPG', 'LFBO', 'EGPH', 'EHAM', 'EBBR', 'LEMD', 'LEBL', 'LPPT', 'LIRF',
            'LIMC', 'LSZH', 'EDDM', 'EDFH', 'EDDT', 'EKCH', 'ENGM', 'ESSA', 'EFHK', 'LOWW']

def get_rains(code):
    arr = np.load("data/rain.npy")
    idx = airports.index(code)
    return arr[:, idx].astype(np.int32)

def get_era_full(param, level):
    arr = np.load("data/{}{}_uint8.npy".format(param, level))
    return arr.astype(np.float32) / 256.

def train_model(airport):
    # Import data
    params = ["z", "z", "z"]
    levels = [500, 850, 1000]

    in1_var = get_era_full(params[0], levels[0])
    in2_var = get_era_full(params[1], levels[1])
    in3_var = get_era_full(params[2], levels[2])

    x = np.concatenate((np.expand_dims(in1_var, axis=3), np.expand_dims(in2_var, axis=3), np.expand_dims(in3_var, axis=3)), axis=3)

    X = np.zeros((13141, 80, 120, 8, 3))
    for i in range(13141):
        X[i,:,:,:,:] = np.rollaxis(x[i:i+8, :, :, :],0,3)

    Y = get_rains(airport)[7:]
    b = np.zeros((Y.shape[0], 2))
    b[np.arange(Y.shape[0]), Y] = 1

    model = None
    if os.path.isfile('model_3d_{}.h5'.format(airport)):
        model = load_model('model_3d_{}.h5'.format(airport))
    else:
        model = Sequential()
        model.add(Convolution3D(128, (3, 3, 3), padding='same', activation='relu', name='block1_conv1', input_shape=(80,120,8,3)))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

        model.add(Convolution3D(256, (3, 3, 3), padding='same', activation='relu', name='block2_conv1'))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

        model.add(Flatten())
        model.add(Dense(2, activation = 'softmax', name='final_fully_connected'))

        adagrad = Adagrad(lr=0.0002)
        model.compile(loss='categorical_crossentropy', optimizer = adagrad, metrics=['accuracy'])

    csv_logger = CSVLogger('{}.log'.format(airport))
    model.fit(X, b, batch_size=20, epochs=100, verbose=1, validation_split=0.2, callbacks=[csv_logger])

    model.save('model_3d_{}.h5'.format(airport))

if __name__ == '__main__':
    airport = sys.argv[1]
    train_model(airport)
