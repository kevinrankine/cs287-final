import numpy as np
import h5py
import string
import re

CORPUS = 'data/text_tokenized.txt'
EMBEDDINGS = 'data/vector/vectors_pruned.200.txt'
TRAIN = 'data/train_random.txt'
DEV = 'data/dev.txt'

def load_corpus(filename, word_dict):
    corpus = [[] for _ in range(523751)]
    with open(filename) as f:
        for line in f:
            line = line.split('\t')
            index = int(line[0])
            title = map(lambda x : word_dict[x],
                        filter(lambda x : x in word_dict,
                        line[1][:-1].split(' ')))
            corpus[index] = title
            
    max_length = max(map(len, corpus))
    corpus = map(lambda x : [word_dict['START']] * (max_length - len(x)) + x, corpus)
    
    return np.array(corpus, dtype=np.int64)

def load_words(filename):
    word_dict = {}
    embeddings = []
    
    with open(filename) as f:
        for index, line in enumerate(f):
            line = map(lambda x : x.strip(), line.split(' '))[:-1]
            word, embedding = line[0], map(float, line[1:])
            
            word_dict[word] = index + 1
            embeddings.append(embedding)
            
    word_dict['START'] = len(word_dict) + 1
    embeddings.append(np.zeros(200))
    
    return word_dict, np.array(embeddings, dtype=np.float32)

def load_training(filename, dev=False):
    qs = []
    ps = []
    Qs = []
    with open(filename, 'r') as f:
        for line in f:
            if dev:
                q, p, Q, _ = line.split('\t')
            else:
                q, p, Q = line.split('\t')
            
            try:
                q = [int(q)]
                p = map(int, p.split(' '))
                Q = map(int, Q.split(' '))
            except(ValueError):
                continue

            qs.append(q)
            ps.append(p)
            Qs.append(Q)
            
    max_length = max(map(len, ps))
    ps = map(lambda x : x + [0] * (max_length - len(x)), ps)
    
    return np.array(qs, dtype=np.int64), \
        np.array(ps, dtype=np.int64), \
        np.array(Qs, dtype=np.int64)

def structure_training(corpus, qs, ps, Qs):
    Xq = []
    Xp = []
    y = []
    for i, q in enumerate(qs):
        for k, pp in enumerate(ps[i]):
            if (pp == 0):
                break
            Xq.append(corpus[q[0]])
            Xp.append(corpus[pp])
            y.append(-1)
        
            for j, p in enumerate(Qs[i]):
                Xq.append(corpus[q[0]])
                Xp.append(corpus[p])
                y.append(1)
        
    return np.array(Xq, dtype=np.int64), np.array(Xp, dtype=np.int64), np.array(y, dtype=np.int64)

if __name__ == '__main__':
    word_dict, embeddings = load_words(EMBEDDINGS)
    corpus = load_corpus(CORPUS, word_dict)
    qs, ps, Qs = load_training(TRAIN)
    dev_qs, dev_ps, dev_Qs = load_training(DEV, dev=True)
    Xq, Xp, y = structure_training(corpus, qs, ps, Qs)
    
    print "corpus shape: ", corpus.shape
    print "qs, ps, Qs shapes: ", qs.shape, ps.shape, Qs.shape
    print "dev_qs, dev_ps, dev_Qs shapes: ", dev_qs.shape, dev_ps.shape, dev_Qs.shape
    print "Xq, Xp, y shapes: ", Xq.shape, Xp.shape, y.shape

    with h5py.File('data/data.hdf5', 'w') as f:
        f['embeddings'] = embeddings
        f['corpus'] = corpus
        f['qs'] = qs
        f['ps'] = ps
        f['Qs'] = Qs
        f['dev_qs'] = dev_qs
        f['dev_ps'] = dev_ps
        f['dev_Qs'] = dev_Qs
        f['Xq'] = Xq
        f['Xp'] = Xp
        f['y'] = y
        

    
    
