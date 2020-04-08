import json
from collections import Counter
import nltk 
from nltk import word_tokenize
import spacy
import re
import pandas as pd
from nltk.tokenize import word_tokenize

#nlp = spacy.load("en_core_web_sm")

answers = set()
count = 100
with open('C:\\squad_v1.1_dataset_training.json', 'r') as f:
    data = json.load(f)
    for i in range(0, len(data['data'])):
        question = (data['data'][i]['question'])
        s = pd.Series(data['data'][i]['paragraph'])
        if len(s) == 1:  
            count = 0
        else:
            for j in range(0, len(s)-1):
                print (i)
                if len(s) <= 1:
                    count = 0
                    print(s[j])
                else:
                    list_comp = set(s[j].split())&set(question.split())
                    list_temp = set(s[j+1].split())&set(question.split())
                    if len(list_comp) > len(list_temp):
                        count = j
                    else:
                        count = j+1 
        answers.add(s[count])
for i in answers:
    print (i)
