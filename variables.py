VOCAB = {} #key = word, value = number of documents word appears in
VECTOR = [] #list of vocab words

PATHS = {'LOGS':'logs/{}.txt',
        'TEST':'datasets/test/',
        'TRAIN':'datasets/train/{}/',
        'VALID':'datasets/valid/{}/'}
    
SIZES = {'TRAIN':1000, #for oversampling
        'VALID':100}  #for oversampling

TOTALS = {'TRAIN':5659,
        'VALID':611}

LABELS = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I'}

IGNORED_FILES = ['.DS_Store'] #for on macOS
