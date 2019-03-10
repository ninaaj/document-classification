BATCH_SIZE, EPOCHS, VERBOSE = 64, 1, 2

VOCAB,VECTOR = {},[] 

PATHS = {'LOGS':'logs/{}.txt',
        'MODELS':'models/{}.h5',
        'TEST':'datasets/test/',
        'TRAIN':'datasets/train/{}/',
        'VALID':'datasets/valid/{}/'}
    
SIZES = {'TEST':2916,
        'TRAIN':490,
        'VALID':5}

TOTALS = {'TEST':2916,
        'TRAIN':5659,
        'VALID':611}

LABELS = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I'}

IGNORED_FILES = ['.DS_Store']

MODEL_NAME = PATHS['MODELS'].format('keras_model')

LOG_NAME = PATHS['LOGS'].format('doc_log_')

CLASS_SIZES = {'TRAIN':{'A':716,'B':656,'C':492,'D':597,'E':511,'F':519,'G':956,'H':567,'I':645},
                'VALID':{'A':45,'B':74,'C':100,'D':97,'E':100,'F':79,'G':13,'H':98,'I':5}
                }

