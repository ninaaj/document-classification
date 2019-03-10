import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from load_data import load_training_data, load_validation_data
from output import *
from variables import *


def get_training_data():
    print('getting training data')
    x_train, y_train = load_training_data()
    Y_train = np_utils.to_categorical(y_train, len(LABELS)) 
    X_val, Y_val, y_val = get_validation_data()
    return x_train, Y_train, y_train, X_val, Y_val, y_val

def get_validation_data():
    print('getting validation data')
    x_valid, y_valid = load_validation_data()
    Y_valid = np_utils.to_categorical(y_valid, len(LABELS))
    return x_valid, Y_valid, y_valid

def train_models_and_classifiers():
    x_train, y_train, og_y_train, x_val, y_val, og_y_val = get_training_data()
    
    print('training models and classifiers')
    
    neural_network(x_train, y_train, og_y_train, x_val, y_val, og_y_val)
    
    naive_bayes('complementNB',ComplementNB(),x_train, og_y_train, x_val, og_y_val)
    
    naive_bayes('multinomialNB',MultinomialNB(),x_train, og_y_train, x_val, og_y_val)
    
    return

def naive_bayes(name, classifier, x_train, y_train, x_val, y_val):
    print(f'training {name} naive bayes classifier')
  
    classifier.fit(x_train, y_train)
    
    score = classifier.score(x_val,y_val)
    
    predictions = [classifier.predict(x_train),classifier.predict(x_val)]

    plot_graphs(name,'training',predictions[0],y_train,490,100,10)
    plot_graphs(name,'validation',predictions[1],y_val,5,1,3)

    print(f'mean accuracy on validation data {score}')
    
    return

def neural_network(x_train, y_train, og_y_train, x_val, y_val, og_y_val):
    print('training sequential neural network')
    
    model = Sequential()
    model.add(Dense(500, input_shape=(len(VOCAB),)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(len(LABELS)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
  
    model.fit(x_train, y_train,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        verbose=VERBOSE,
        validation_data=(x_val, y_val))

    predictions = [model.predict_classes(x_train),model.predict_classes(x_val)]
    
    plot_graphs('sequentialNN','training',predictions[0],og_y_train,490,100,10)
    plot_graphs('sequentialNN','validation',predictions[1],og_y_val,5,1,3)
    
    return


if __name__ == '__main__':
    train_models_and_classifiers()
