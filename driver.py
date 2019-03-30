import os
from variables import *
from model import *

BATCH_SIZE = 128
EPOCHS =  10
VERBOSE = 2
UNITS = 500

make_model(BATCH_SIZE,EPOCHS,VERBOSE,UNITS)