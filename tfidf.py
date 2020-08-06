import MeCab
import os
import numpy as np
import math

files = [i for i in os.listdir('./') if i.endswith('trn1')]
texts = [[] for i in range(len(files))]
out = ""

m = MeCab.Tagger()

dic = {}
tf = np.empty((len(files),0))

for n,file in enumerate(files):
    with open(file, encoding='SHIFT_JIS') as doc:
        for line in doc:
            texts[n].append(line)
            words = m.parse(line).split()
            del words[-1]
            for i in range(0,len(words),2):
                if words[i+1].startswith('名詞'):
                    if words[i] in dic:
                        tf[n][dic[words[i]]] += 1
                    else:
                        dic[words[i]] = len(dic)
                        tf = np.append(tf,np.zeros([len(files),1]),axis=1)
                        tf[n][dic[words[i]]] += 1

for i in range(len(dic)):
    df = sum(j>0 for j in tf[:,i])
    idf = round(math.log10(len(files)/df)+1,3) 
    tf[:,i] *= idf
    
for n,text in enumerate(texts):
    scores = {}
    for j,line in enumerate(text):
        words =  m.parse(line).split()
        del words[-1]
        ids = []
        for i in range(0,len(words),2):
           if(words[i+1].startswith('名詞')):
               if dic[words[i]] not in ids:
                   ids.append(dic[words[i]])
        tmp = 0
        for i in ids:
            tmp += tf[n][i]
        scores[j] = round(tmp,3)
    scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    
    out += '-----'+files[n]+'-----\n'
    for i in range(5):
        out += str(i+1)+'位:('+str(scores[i][0]+1)+'行目)'+text[scores[i][0]]+'\n'

with open('tfidf-result.txt','w') as txt:
    txt.write(out)          