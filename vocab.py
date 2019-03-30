import os, re
from variables import *

regex = re.compile('[^a-zA-Z]')

def build_vocab():
    print('building vocabulary')
    for label in LABELS:
        print(f'processing files for class {LABELS[label]}')
        process_files(PATHS['TRAIN'].format(LABELS[label]))
    VECTOR.extend(list(VOCAB.keys()))
    VECTOR.sort()
    print(f'\n\nvocabulary size = {len(VOCAB)}')

def process_files(path):

    for filename in os.listdir(path):
        if filename in IGNORED_FILES:
            continue

        with open(path+filename, encoding="utf8", errors='ignore') as file:
            file = file.read()
            file = file.split('\n')
            prev = []
            for line in file: 
                line = line.split(' ')
                line[0] = regex.sub('', line[0])
                
                if len(line) != 2 or not line[0].isalpha():
                    continue 
                
                if line[0] in prev:
                    continue

                if line[0] not in VOCAB:
                    VOCAB[line[0]] = 0

                VOCAB[line[0]] += 1    
                prev.append(line[0])
    rare = []
    
    for word in VOCAB:
        if VOCAB[word] < 2:
            rare.append(word)
    
    for word in rare:
        VOCAB.pop(word)

def get_index(word):
    return VECTOR.index(word) 
    
build_vocab()
