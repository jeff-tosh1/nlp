import MeCab
import os
import numpy as np
import math

files = [i for i in os.listdir('./') if i.endswith('trn1')]
texts = []

m = MeCab.Tagger()

nouns = {}
tf = np.empty((len(files),0))

for n,file in enumerate(files):
    with open(file, encoding='Shift-JIS') as f:
        text = f.read()
    words = m.parse(text).split()
    del words[-1]
    
    for i in range(0,len(words),2):
        if(words[i+1].startswith('名詞')):
            if(words[i] in nouns):
                tf[n][nouns[words[i]]] += 1
            else:
                nouns[words[i]] = len(nouns)
                tf = np.append(tf,np.zeros([len(files),1]),axis=1)
                tf[n][nouns[words[i]]] += 1
            
for i in range(len(nouns)):
    df = sum(j>0 for j in tf[:,i])
    idf = round(math.log10(len(files)/df)+1,3) 
    tf[:,i] *= idf

scores = {}
with open(files[0], encoding='Shift-JIS') as f:
    texts.extend(f.read().split('\n'))
    for n,line in enumerate(texts):
       words =  m.parse(line).split()
       del words[-1]
       
       ids = []
       for i in range(0,len(words),2):
           if(words[i+1].startswith('名詞')):
               if nouns[words[i]] not in ids:
                   ids.append(nouns[words[i]])
       tmp = 0
       for i in ids:
           tmp += tf[0][i]
       scores[n] = round(tmp,3)

scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
print(scores)
