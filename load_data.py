import os, re
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import numpy as np

from random import shuffle
from variables import *
from vocab import get_index

regex = re.compile('[^a-zA-Z]')

def get_data(path,total):
    
    current_class = []
    
    for filename in os.listdir(path):
        
        if filename in IGNORED_FILES:
            continue
    
        with open(path+filename, encoding='utf8', errors='ignore') as file:
            
            file = file.read()
            file = file.split('\n')
            
            bag = np.zeros((len(VECTOR),), dtype=np.float64)  
            
            terms = 0
            
            for line in file:   
                line = line.split(' ')
                line[0] = regex.sub('', line[0]) 
                
                if len(line) != 2 or not line[0].isalpha():
                    continue 
                
                if line[0] in VECTOR:
                    bag[get_index(line[0])] = int(line[1])
                
                terms += int(line[1])
            
            # weight bag using TF-IDF weighting
            # TF-IDF = # of term occurances in document / total terms in document * natural log of total # of documents / # of documents term occurs in
            bag = [(bag[i]/terms)*np.log(total/VOCAB[VECTOR[i]]) for i in range(0,len(bag))]
            
            current_class.append(bag)
    
    return current_class

def load_training_data():
    
    train_data, train_labels = [], []
    
    for label in LABELS:
        print(f'\nloading training data for class {LABELS[label]}')
        path = PATHS['TRAIN'].format(LABELS[label])
        data = get_data(path,TOTALS['TRAIN'])
        train_data.extend(data)
        train_labels.extend([label for _ in range(0,len(data))])
    
    train = list(zip(train_data,train_labels))
    shuffle(train)
    train_data, train_labels = zip(*train)

    return np.array(train_data), np.array(train_labels)

def load_validation_data():
    
    valid_data, valid_labels = [], [] 
    
    for label in LABELS:
        print(f'\nloading validation data for class {LABELS[label]}')
        path = PATHS['VALID'].format(LABELS[label])
        data = get_data(path,TOTALS['TRAIN'])
        valid_data.extend(data)
        valid_labels.extend([label for _ in range(0,len(data))])
    
    valid = list(zip(valid_data,valid_labels))
    shuffle(valid)
    valid_data, valid_labels = zip(*valid)

    return np.array(valid_data), np.array(valid_labels)
