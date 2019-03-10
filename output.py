import matplotlib.pyplot as plt
import numpy as np
from variables import LABELS, LOG_NAME

def plot_graphs(classifier,dataset,predictions,labels,total,ticks,top):
    correct= {'A':0,'B':0,'C':0,'D':0,'E':0,'F':0,'G':0,'H':0,'I':0}
   
    for i in range(0,len(labels)):
        if labels[i] == predictions[i]:
            correct[LABELS[labels[i]]] += 1  
    
    incorrect = [(total - c) for c in list(correct.values())]
    
    corr_arr = np.array(list(correct.values()))
    incorr_arr = np.array(incorrect)

    classes = np.array(list(correct.keys()))
    
    fig = plt.figure()
    
    ind = np.arange(len(classes))    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, corr_arr, width, color='#a44b4b')
    p2 = plt.bar(ind, incorr_arr, width,
                bottom=corr_arr, color='#2c7ac4')

    plt.ylabel('number of predictions')
    plt.title(f'{classifier} ({dataset})')
    plt.xticks(ind, classes)
    plt.yticks(np.arange(0, (total+top), ticks))
    plt.legend((p1[0], p2[0]), ('correct', 'incorrect'))
    fig.savefig(f'{classifier}_{dataset}.png')

def print_log(text):
    file = open(LOG_NAME, 'a')
    file.write(str(text) + '\n')
    print(text)
