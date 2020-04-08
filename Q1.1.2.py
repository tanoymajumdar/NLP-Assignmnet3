import pandas as pd
import readability as ra
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import scipy.stats
from nltk import word_tokenize
from nltk import bigrams
from collections import Counter


df = pd.read_csv("C://training_dataTry.csv")
df2 = pd.read_csv("C://testing_data.csv")

readability_score = []
unigramScore = []
bigramScore = []
AA = []
ListofWords =[]
unigramScoreHigh = []
bigramScoreHigh = []

for i in df['summary']:
    ListofWords.append(i)
    toks = word_tokenize(i)
    A = Counter(toks).most_common(1)
    unigramScoreHigh.append(A[0][1])
    B = Counter(bigrams(toks)).most_common(1)
    bigramScoreHigh.append(B[0][1])

for i in df['summary']:
    result = ra.getmeasures(i, lang = 'en')
    readability_score.append(result['readability grades']['FleschReadingEase'])
    A.append(i)
countUni = 0
countBi = 0
for i in AA:
    Words = i.split()
    for j in range(1, len(Words)):
        if (Words[j] == Words[j-1]):
            countUni = countUni +1
    for m in range(3, len(Words)):
        if (Words[m]==Words[m-2] and Words[m-1] == Words[m-3]):
            countBi = countBi + 1
    unigramScore.append(countUni)
    bigramScore.append(countBi)
    countUni = 0
    countBi = 0

df['ra score'] = readability_score
df['uni Score'] = unigramScore
df['bi score'] = bigramScore
df['uni Score H'] = unigramScoreHigh
df['bi score H'] = bigramScoreHigh


readability_score2 = []
unigramScore2 = []
bigramScore2 = []
AA2 = []
ListofWords2 = []
unigramScore2High = []
bigramScore2High = []

for i in df['summary']:
    ListofWords2.append(i)
    toks = word_tokenize(i)
    A = Counter(toks).most_common(1)
    unigramScore2High.append(A[0][1])
    B = Counter(bigrams(toks)).most_common(1)
    bigramScore2High.append(B[0][1])

for i in df2['summary']:
    result = ra.getmeasures(i, lang = 'en')
    readability_score2.append(result['readability grades']['FleschReadingEase'])
    A2.append(i)
countUni2 = 0
countBi2 = 0
for i in AA2:
    Words = i.split()
    for j in range(1, len(Words)):
        if (Words[j] == Words[j-1]):
            countUni2 = countUni2 +1
    for m in range(3, len(Words)):
        if (Words[m]==Words[m-2] and Words[m-1] == Words[m-3]):
            countBi2 = countBi2 + 1
    unigramScore2.append(countUni)
    bigramScore2.append(countBi)
    countUni2 = 0
    countBi2 = 0

df2['ra score'] = readability_score2
df2['uni Score'] = unigramScore2
df2['bi score'] = bigramScore2
df['uni Score H'] = unigramScore2High
df['bi score H'] = bigramScore2High

X = df[df.columns[-3:df.columns.size]]
Y = df['grammaticality']
clf = linear_model.LogisticRegression(C=1e40, solver='newton-cg')
fitted_model = clf.fit(X, Y)
prediction_result = clf.predict(df2[df.columns[-5:df.columns.size]])
finalMSE = mean_squared_error(df2['grammaticality'], prediction_result)
finalP = scipy.stats.pearsonr(df2['grammaticality'], prediction_result)
print(finalP)
print(finalMSE)