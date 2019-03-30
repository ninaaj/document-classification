import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from load_data import load_training_data, load_validation_data
from variables import *

def get_training_data():
    print('\ngetting training data')
    x_train, y_train = load_training_data()
    Y_train = np_utils.to_categorical(y_train, len(LABELS))
    X_val, Y_val = get_validation_data()
    return x_train, Y_train, X_val, Y_val

def get_validation_data():
    print('\ngetting validation data')
    x_valid, y_valid = load_validation_data()
    Y_valid = np_utils.to_categorical(y_valid, len(LABELS))
    return x_valid, Y_valid

def make_model(batch_size,epochs,verbose,units):
    print(f'\n\nmodel units {units} batch_size {batch_size}')
    x_train, y_train, x_val, y_val = get_training_data()
    
    model_name = f'{batch_size}_{units}'
    model_path = model_name + '_{epoch:02d}.h5'
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=0, verbose=2),ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)]
    
    model = Sequential()
    
    model.add(Dense(units, input_shape=(len(VOCAB),),activation='relu'))                        
    model.add(Dropout(0.5))
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(LABELS), activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
  
    model.fit(x_train, y_train,
        batch_size=batch_size, 
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
        validation_data=(x_val, y_val))
    
    print(model.summary())

    loss, accuracy = model.evaluate(x_val, y_val, verbose=2)

    print(f'\n\nTest Loss {loss}')
    print(f'Test Accuracy {accuracy}')
    
    return
