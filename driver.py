from model import make_model

BATCH_SIZE = 128 #number of samples per gradient update 
EPOCHS =  10 #number of epochs to train the model (may be less due to early stopping callback)
VERBOSE = 2 #verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch
UNITS = 500 #dimensionality of the output space

make_model(BATCH_SIZE,EPOCHS,VERBOSE,UNITS)
