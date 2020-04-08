import json
from collections import Counter
import nltk 
from nltk import word_tokenize
import spacy
import re
import numpy as np
import matplotlib.pyplot as pyplot

nlp = spacy.load("en_core_web_sm")

question_list = []

with open('C:\\squad_v1.1_dataset_training.json', 'r') as f:
    data = json.load(f)
    print(len(data['data']))
    for i in range(0, len(data['data'])):
        question = data['data'][i]['question'].split()
        question_list.append(question[0])
    print (type(Counter(question_list)))
    #counts,bins=np.histogram(Counter(question_list))
    #pyplot.hist(bins[:-1],bins,weights=counts)
    pyplot.bar(Counter(question_list).keys(),Counter(question_list).values())
    pyplot.show()