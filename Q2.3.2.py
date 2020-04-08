import json
from collections import Counter
import nltk 
from nltk import word_tokenize
import spacy
import re
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import readability as ra
import nltk 
from nltk import word_tokenize
from nltk import bigrams
from nltk import trigrams
from collections import Counter
import numpy as np
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
from gensim import models
from scipy import spatial
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

#nlp = spacy.load("en_core_web_sm")
model = models.KeyedVectors.load_word2vec_format("C:\GoogleNews-vectors-negative300.bin.gz", binary=True, limit = 3000)
answers = set()
count = 100
questions = []
with open('C:\\squad_v1.1_dataset_training.json', 'r') as f:
    data = json.load(f)
    for i in range(0, len(data['data'])):
        question = (data['data'][i]['question'])
        questions.append(question)
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

fin_array = np.zeros((1,300))
fin_arrayQ = np.zeros((1,300))
temp_arraylist = []
temp_arraylistQ = []
cosine_list = []
cosine_listFinal = []


for i in answers:
    for words in i:
        count = count + 1
        if words in model:
            vector = model.wv[words]
            fin_array = np.add(fin_array, vector)
        else:
            continue
    fin_array = np.true_divide(vector, count)
    temp_arraylist.append(fin_array)
for i in questions:
    for words in i:
        count = count + 1
        if words in model:
            vector = model.wv[words]
            fin_array = np.add(fin_array, vector)
        else:
            continue
    fin_arrayQ = np.true_divide(vector, count)
    temp_arraylistQ.append(fin_array)
for nn in range(0, len(temp_arraylist)):
    for kk in range(0, len(temp_arraylist)):
        result = 1 - spatial.distance.cosine(nn, kk)
        cosine_list.append(result)
    cosine_list = cosine_list.sort()
    cosine_listFinal.append(cosine_list[0])
    cosine_list.clear()
print(cosine_list)