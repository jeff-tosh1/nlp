import MeCab
import os
import numpy as np
import math
import networkx as nx

files = [i for i in os.listdir('./') if i.endswith('trn1')]
m = MeCab.Tagger()
out = ""

for f in files:
    text = []
    dic = {}
    tf = np.empty((0,len(dic)))
    with open(f, encoding='Shift-JIS') as file:
        for n,line in enumerate(file):
            text.append(line)
            words = m.parse(line).split()
            del words[-1]
            
            tf = np.append(tf,np.zeros([1,len(dic)]),axis=0)
            for i in range(0,len(words),2):
                if words[i+1].startswith('名詞'):
                    if words[i] in dic:
                        tf[n][dic[words[i]]] += 1
                    else:
                        dic[words[i]] = len(dic)
                        tf = np.append(tf,np.zeros([n+1,1]),axis=1)
                        tf[n][dic[words[i]]] += 1
                        
    for i in range(len(dic)):
        df = sum(j>0 for j in tf[:,i])
        idf = round(math.log10(len(tf)/df)+1,3) 
        tf[:,i] *= idf
        
    sim_mat = np.empty((len(tf),len(tf)))
    for i in range(len(tf)):
        for j in range(len(tf)):
            if np.linalg.norm(tf[i]) * np.linalg.norm(tf[j]) == 0:
                sim_mat[i][j] = 0
            else:
                sim_mat[i][j] = np.dot(tf[i], tf[j]) / (np.linalg.norm(tf[i]) * np.linalg.norm(tf[j]))
        
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    
    scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    
    out += '-----'+f+'-----\n'
    for i in range(5):
        out += str(i+1)+'位:('+str(scores[i][0]+1)+'行目)'+text[scores[i][0]]+'\n'
    
with open('textrank-result.txt','w') as file:
    file.write(out)