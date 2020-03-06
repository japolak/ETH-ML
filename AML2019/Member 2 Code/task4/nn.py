from main import read_test_data, read_data, StandardScaler, dir_cur
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape, Conv2D, MaxPooling2D, Flatten
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, to_categorical
from time import gmtime, strftime, localtime, time
import argparse
import sys


def lstm():
    model = Sequential()
    return model


def cnn_paper():
    # implemented the model here
    # https://iopscience.iop.org/article/10.1088/1741-2552/aadc1f/pdf
    # create model
    model = Sequential()
    # feature extraction
    model.add(Reshape((8, 900, 1), input_shape=(7200, 1)))

    model.add(Conv2D(3, (1, 10), padding='same', activation='relu'))

    model.add(Conv2D(3, (3, 1), activation='relu'))
    model.add(MaxPooling2D((2, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(5, (1, 5), padding='same', activation='relu'))

    model.add(Conv2D(5, (3, 1), activation='relu'))
    model.add(MaxPooling2D((1, 5), strides=(1, 3), padding='same'))

    model.add(Conv2D(7, (1, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D((1, 6), strides=(1, 4)))

    model.add(Conv2D(10, (1, 37)))

    model.add(Conv2D(3, (1, 1)))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


def cnn():

    model = Sequential()
    # feature extraction
    # model.add(Reshape((1536, 1), input_shape=(1536,)))
    # model.add(Reshape((3, 512, 1), input_shape=(1536, 1)))
    model.add(Reshape((3, 512, 1), input_shape=(1536,)))

    model.add(Conv2D(3, (1, 10), padding='same', data_format='channels_first',
                     activation='relu', input_shape=(3, 512, 1)))

    model.add(Conv2D(3, (3, 1), activation='relu'))
    model.add(MaxPooling2D((2, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(5, (1, 4), activation='relu', padding='same'))

    model.add(Conv2D(5, (1, 1), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(1, 4), padding='same'))

    model.add(Conv2D(7, (1, 4), padding='same', activation='relu'))
    model.add(MaxPooling2D((1, 6), strides=(1, 4)))

    model.add(Conv2D(10, (1, 15)))

    model.add(Conv2D(3, (1, 1)))

    model.add(Flatten())
    # model.add(Dense(3, activation='softmax'))
    model.add(Dense(3, activation='softmax'))  # , input_shape=(3,1)))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    print('Starting at   :', strftime("%Y-%m-%d %H:%M:%S", localtime()))
    t0 = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='cnn',
                        help="Enter the model")
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help="Enter the number of epochs")
    parser.add_argument('-s', '--store', default=True, action='store_true',
                        help="Store the predicted results")
    parser.add_argument('-d', '--dry', default=False, action='store_true',
                        help="Show the model summary only")
    parser.add_argument('-v', '--valid', default=None, type=float,
                        help="Validation size")
    args = parser.parse_args()

    if args.model == 'cnn':
        print("Fitting CNN\n")
        model = cnn()
    elif args.model == 'paper':
        print("Fitting CNN model from paper\n")
        sys.exit()
    else:
        sys.exit("Model not found!\n")

    if args.dry:
        print('Model: {}'.format(args.model))
        print('Epoch: {}'.format(args.epochs))
        model.summary()
        sys.exit()

    X, y = read_data()
    X = StandardScaler().fit_transform(X)
    # X = X.reshape((-1, 3, 512, 1))
    y = y.reshape((-1, 1))
    y = OneHotEncoder(sparse=False).fit_transform(y)

    # sparse=True: (0, 1)
    # sparse=True: [ 0.  1.  0.]
    # y = to_categorical(y) # this return [ 0.  0.  1.  0.] for some reasons

    if not args.valid:
        model.fit(X, y, verbose=True, epochs=args.epochs)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.valid, shuffle=True)

        model.fit(X_train, y_train, verbose=True, epochs=args.epochs,
                validation_data=(X_val, y_val))

    X_test = read_test_data()
    X_test = StandardScaler().fit_transform(X_test)
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)
    pred += 1
    print(np.unique(pred, return_counts=1))
    np.savetxt("{}/submission_test.csv".format(dir_cur),
               np.dstack((np.arange(0, pred.size), pred))[0],
               '%i,%i', comments='', header="Id,y")

    print('Ending at     :', strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print('Total runtime : {:.3f} sec\n'.format(time() - t0))
