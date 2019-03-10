import os, re
from numpy import array, float64, log, zeros
from random import sample, shuffle
from variables import *
from vocab import get_index

regex = re.compile('[^a-zA-Z]')

def get_data(path,size,total):
    current_class = []
    for filename in os.listdir(path):
        if filename in IGNORED_FILES:
            continue
        with open(path+filename, encoding="utf8", errors='ignore') as file:
            file = file.read()
            file = file.split('\n')
            bag, terms = zeros((len(VECTOR),), dtype=float64), 0
            
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
            bag = [(bag[i]/terms)*log(total/VOCAB[VECTOR[i]]) for i in range(0,len(bag))]
            
            current_class.append(bag)
    
    return sample(current_class,size)

def load_training_data():
    train_data, train_labels = [], []
    
    for label in LABELS:
        print(f'loading training data for class {LABELS[label]}')
        path = PATHS['TRAIN'].format(LABELS[label])
        train_data.extend(get_data(path,SIZES['TRAIN'],TOTALS['TRAIN']))
        train_labels.extend([label for _ in range(0,SIZES['TRAIN'])])
    
    #shuffle the data and labels 
    train = list(zip(train_data,train_labels))
    shuffle(train)
    train_data, train_labels = zip(*train)
    return array(train_data), array(train_labels)

def load_validation_data():
    valid_data, valid_labels = [], []
    
    for label in LABELS:
        print(f'loading validation data for class {LABELS[label]}')
        path = PATHS['VALID'].format(LABELS[label])
        valid_data.extend(get_data(path,SIZES['VALID'],TOTALS['TRAIN']))
        valid_labels.extend([label for _ in range(0,SIZES['VALID'])])
    
    #shuffle the data and labels 
    valid = list(zip(valid_data,valid_labels))
    shuffle(valid)
    valid_data, valid_labels = zip(*valid)
    return array(valid_data), array(valid_labels)

