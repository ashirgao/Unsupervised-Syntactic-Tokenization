
##### Import libraries


```python
import os
import random
import multiprocessing

import numpy as np
from tqdm import tqdm 
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
```


```python
NUM_PROCESSES = multiprocessing.cpu_count() - 1
```

###### Read already downloaded data. (Find links to corpus below)


```python
# http://www.statmt.org/wmt11/training-monolingual-news-2011.tgz
# http://www.statmt.org/wmt11/training-monolingual-news-2010.tgz
corpus1 = open("data/training-monolingual/news.2011.en.shuffled","r").read()
corpus2 = open("data/training-monolingual/news.2010.en.shuffled","r").read()
corpus = corpus1 + "\n" + corpus2
```

###### Preprocess


```python
sentences = corpus.split("\n")

for i,v in enumerate(sentences):
    sentences[i] = v.replace(",","").replace(".","").replace("!","").replace("?","").replace("/","").replace("'","").lower()
print("Total sentences : {}".format(len(sentences)))
```

    Total sentences : 20142184


###### Train CountVectorizer and generate dictionary (and also see how to generate a reverse one)


```python
vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
```


```python
vectorizer = vectorizer.fit(sentences)
print("Length of dictionary : {}".format(len(vectorizer.vocabulary_)))
```

    Length of dictionary : 1050336



```python
word2index = vectorizer.vocabulary_
index2word = dict(zip(word2index.values(),word2index.keys()))
```

###### Generate a co-occurance matriz


```python
# unordered_co_mat = (sentences_mat.T * sentences_mat) 
# unordered_co_mat.setdiag(0) 
# print("Shape of co-occurance matrix (unordered) - {}".format(unordered_co_mat.shape))
# list(word2index.keys())[0:1000]
```

##### Function that generates position aware co-occurance matrix


```python
def create_dist_map(dist):
    """ Creates co-occurance matrix of size(vocab_len, vocal_len) between all elements
        at given distance "dist" in corpus
    """
    # initialize matrix to zeros
    dist_mat = sparse.lil_matrix((len(word2index),len(word2index)),dtype=np.int32)
    
    # train on random fixed length subset of corpus
#     for sentence in tqdm(random.sample(sentences,100000)):
    # OR
    # train on all corpus
    for sentence in tqdm(sentences):

        words = list(filter(lambda a: a != "", sentence.split(" ")))
        for i,v in enumerate(words[:-dist]): 
            try:
                dist_mat[word2index[v],word2index[words[i+dist]]] += 1
            except:
                # fails if word not present in dictionary
                # do nothing
                pass

        pass
    return(dist_mat)
```

###### Single process method


```python
# co_oc1 = create_dist_map(1)
# co_oc2 = create_dist_map(2)
# co_oc3 = create_dist_map(3)
# co_oc4 = create_dist_map(4)
# co_oc5 = create_dist_map(5)
```

###### Multi-processing method


```python
pool = multiprocessing.Pool(processes=NUM_PROCESSES)  
co_oc1, co_oc2,co_oc3,co_oc4,co_oc5 = pool.map(create_dist_map,[1,2,3,4,5])
pool.close()
pool.join()
```

    100%|██████████| 20142184/20142184 [1:34:38<00:00, 3546.79it/s]  
     90%|████████▉ | 18118721/20142184 [1:39:29<10:13, 3299.26it/s]
    100%|██████████| 20142184/20142184 [1:44:30<00:00, 3212.00it/s]
    100%|██████████| 20142184/20142184 [1:48:25<00:00, 3096.03it/s]
    100%|██████████| 20142184/20142184 [1:51:41<00:00, 3005.82it/s]


###### Save matrices


```python
sparse.save_npz("co_oc_matrices/co_oc_1",co_oc1.tocoo())
sparse.save_npz("co_oc_matrices/co_oc_2",co_oc2.tocoo())
sparse.save_npz("co_oc_matrices/co_oc_3",co_oc3.tocoo())
sparse.save_npz("co_oc_matrices/co_oc_4",co_oc4.tocoo())
sparse.save_npz("co_oc_matrices/co_oc_5",co_oc5.tocoo())
```

###### Load previously saved matrices
- to compare is saved and loaded matrices are the same do 

 ``` (saved_sparse_mat != loaded_sparse_matrix)``` 
 
 and check for elements present in the newly generated sparse matrix


```python
# co_oc1 = sparse.load_npz("co_oc_matrices/co_oc1.npz")
# co_oc2 = sparse.load_npz("co_oc_matrices/co_oc2.npz")
# co_oc3 = sparse.load_npz("co_oc_matrices/co_oc3.npz")
# co_oc4 = sparse.load_npz("co_oc_matrices/co_oc4.npz")
# co_oc5 = sparse.load_npz("co_oc_matrices/co_oc5.npz")
```

###### Find compund tokens in any given sentence


```python
def inference(test):
    words = test.lower().split()
    words = [i for i in words if i in word2index.keys()]
    
    l0 = [ 0.0 for i in words]
    l1 = [ co_oc1[word2index[v],word2index[words[i+1]]] for i,v in enumerate(words[:-1])]
    l2 = [ co_oc2[word2index[v],word2index[words[i+2]]] for i,v in enumerate(words[:-2])]
    l3 = [ co_oc3[word2index[v],word2index[words[i+3]]] for i,v in enumerate(words[:-3])]
    l4 = [ co_oc4[word2index[v],word2index[words[i+4]]] for i,v in enumerate(words[:-4])]
    l5 = [ co_oc5[word2index[v],word2index[words[i+5]]] for i,v in enumerate(words[:-5])]
    


    row = "" 
    for i in words:
        row += i+"\t"
    row += "\n"
    for i in l0:
        row += str(i)+"\t\t"
    row += "\n\t"
    for i in l1:
        row += str(i)+"\t\t"
    row += "\n\t\t"
    for i in l2:
        row += str(i)+"\t\t"
    row += "\n\t\t\t"
    for i in l3:
        row += str(i)+"\t\t"
    row += "\n\t\t\t\t"
    for i in l4:
        row += str(i)+"\t\t"
    row += "\n\t\t\t\t\t"
    for i in l5:
        row += str(i)+"\t\t"
    row += "\n"
    print(row)
    
    
    print("\n\nSuggestions : ")
    for ind,val in enumerate(l1):
        if(l0[ind]<val and l0[ind+1]<val):
            if(ind==0):
                if(l2[ind]<val):
                    print("\t{} {} - {}".format(words[ind],words[ind+1],val))
            elif(ind==len(l1)-1):
                if(l2[ind-1]<val):
                    print("\t{} {} - {}".format(words[ind],words[ind+1],val))
            else:
                if(l2[ind-1]<val and l2[ind]<val):
                    print("\t{} {} - {}".format(words[ind],words[ind+1],val))
    
    
    for ind,val in enumerate(l2):
        if(l1[ind]<val and l1[ind+1]<val):
            if(ind == 0):
                if(l3[ind]<val):
                    print("\t{} {} {} - {}".format(words[ind],words[ind+1],words[ind+2],val))
            elif(ind==len(l2)-1):
                if(l3[ind-1]<val):
                    print("\t{} {} {} - {}".format(words[ind],words[ind+1],words[ind+2],val))
            else:
                if(l3[ind-1]<val and l3[ind]<val):
                    print("\t{} {} {} - {}".format(words[ind],words[ind+1],words[ind+2],val))
               
                    
    for ind,val in enumerate(l3):
        if(l2[ind]<val and l2[ind+1]<val):
            if(ind == 0):
                if(l4[ind]<val):
                    print("\t{} {} {} {} - {}".format(words[ind],words[ind+1],words[ind+2],words[ind+3],val))
            elif(ind==len(l3)-1):
                if(l4[ind-1]<val):
                    print("\t{} {} {} {} - {}".format(words[ind],words[ind+1],words[ind+2],words[ind+3],val))                
            else:
                if(l4[ind-1]<val and l4[ind]<val):
                    print("\t{} {} {} {} - {}".format(words[ind],words[ind+1],words[ind+2],words[ind+3],val))
                    pass
                    
    for ind,val in enumerate(l4):
        if(l3[ind]<val and l3[ind+1]<val):
            if(ind == 0):
                if(l5[ind]<val):
                    print("\t{} {} {} {} {} - {}".format(words[ind],words[ind+1],words[ind+2],words[ind+3],words[ind+4],val))
                    pass
            elif(ind==len(l4)-1):
                if(l5[ind-1]<val):
                    print("\t{} {} {} {} {} - {}".format(words[ind],words[ind+1],words[ind+2],words[ind+3],words[ind+4],val))
                    pass                
            else:
                if(l5[ind-1]<val and l5[ind]<val):
                    print("\t{} {} {} {} {} - {}".format(words[ind],words[ind+1],words[ind+2],words[ind+3],words[ind+4],val))
                    pass
                
        
```


```python
s = "new york stock exchange is a busy place "
print(s+"\n")
inference(s)
```

    new york stock exchange is a busy place 
    
    new	york	stock	exchange	is	a	busy	place	
    0.0		0.0		0.0		0.0		0.0		0.0		0.0		0.0		
    	236308		2754		6144		277		275157		2626		16		
    		2806		5788		703		1239		382		6708		
    			5848		840		872		0		3853		
    				5436		4100		0		22		
    					15977		6		28		
    
    
    
    Suggestions : 
    	new york - 236308
    	stock exchange - 6144
    	is a - 275157
    	a busy place - 6708
    	new york stock exchange - 5848



```python
s = "South Africa is one token"
print(s+"\n")
inference(s)
```

    South Africa is one token
    
    south		africa		is		one		token		
    0.0		0.0		0.0		0.0		0.0		
    	16986		827		36105		2		
    		1557		109		6		
    			374		0		
    				0		
    					
    
    
    
    Suggestions : 
    	south africa - 16986
    	is one - 36105
