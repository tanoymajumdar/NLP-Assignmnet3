import json
from collections import Counter
import nltk 
from nltk import word_tokenize
import spacy
import re
import matplotlib.pyplot as pyplot
import numpy as np

nlp = spacy.load("en_core_web_sm")

answer_list = []

with open('C:\\squad_v1.1_dataset_training.json', 'r') as f:
    data = json.load(f)
    print(len(data['data']))
    for i in range(0, len(data['data'])):
        answer = nlp(data['data'][i]['answer'])
        for ent in answer.ents:
            answer_list.append(ent.label_)
    print(Counter(answer_list))
    pyplot.bar(Counter(answer_list).keys(),Counter(answer_list).values())
    pyplot.show()