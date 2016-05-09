import numpy as np
import h5py
import string
import re

CORPUS = 'data/text_tokenized.txt'
EMBEDDINGS = 'data/vector/vectors_pruned.200.txt'
TRAIN = 'data/train_random.txt'
DEV = 'data/dev.txt'

def load_corpus(filename, word_dict):
    title_corpus = [[] for _ in range(523751)]
    body_corpus = [[] for _ in range(523751)]
    max_body_length = 100 # make this a parameter
    
    with open(filename) as f:
        for line in f:
            line = line.split('\t')
            index = int(line[0])
            title = map(lambda x : word_dict[x],
                        filter(lambda x : x in word_dict,
                               line[1][:-1].split(' ')))
            body = map(lambda x : word_dict[x],
                       filter(lambda x : x in word_dict,
                              line[2][:-1].split(' ')))[:max_body_length]
            title_corpus[index] = title
            body_corpus[index] = body
            
    max_title_length = max(map(len, title_corpus))
    padding = [word_dict['START'] for i in xrange(max_title_length)]
    title_corpus = map(lambda x : padding[:max_title_length - len(x)] + x, title_corpus)
    padding = [word_dict['START'] for i in xrange(max_body_length)]
    body_corpus = map(lambda x : padding[:max_body_length - len(x)] + x, body_corpus)

    return np.array(title_corpus, dtype=np.int64), np.array(body_corpus, dtype=np.int64)

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

def load_data(filename, dev=False):
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

def structure_data(corpus, qs, ps, Qs):
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
    title_corpus, body_corpus = load_corpus(CORPUS, word_dict)
    
    train_qs, train_ps, train_Qs = load_data(TRAIN)
    dev_qs, dev_ps, dev_Qs = load_data(DEV, dev=True)
    
    title_train_Xq, title_train_Xp, title_train_y = structure_data(title_corpus, train_qs, train_ps, train_Qs)
    title_dev_Xq, title_dev_Xp, title_dev_y = structure_data(title_corpus, dev_qs, dev_ps, dev_Qs)
    body_train_Xq, body_train_Xp, body_train_y = structure_data(body_corpus, train_qs, train_ps, train_Qs)
    body_dev_Xq, body_dev_Xp, body_dev_y = structure_data(body_corpus, dev_qs, dev_ps, dev_Qs)
    
    print "title corpus shape: ", title_corpus.shape
    print "body corpus shape: ", body_corpus.shape
    print "train_qs, train_ps, train_Qs shapes: ", train_qs.shape, train_ps.shape, train_Qs.shape
    print "dev_qs, dev_ps, dev_Qs shapes: ", dev_qs.shape, dev_ps.shape, dev_Qs.shape
    
    print "title_train_Xq, title_train_Xp, title_train_y shapes: ", title_train_Xq.shape, title_train_Xp.shape, title_train_y.shape
    print "title_dev_Xq, title_dev_Xp, title_dev_y shapes: ", title_dev_Xq.shape, title_dev_Xp.shape, title_dev_y.shape
    
    print "body_train_Xq, body_train_Xp, body_train_y shapes: ", body_train_Xq.shape, body_train_Xp.shape, body_train_y.shape
    print "body_dev_Xq, body_dev_Xp, body_dev_y shapes: ", body_dev_Xq.shape, body_dev_Xp.shape, body_dev_y.shape

    with h5py.File('data/data.hdf5', 'w') as f:
        f['embeddings'] = embeddings
        f['title_corpus'] = title_corpus
        f['body_corpus'] = body_corpus
        
        f['train_qs'] = train_qs
        f['train_ps'] = train_ps
        f['train_Qs'] = train_Qs
        
        f['dev_qs'] = dev_qs
        f['dev_ps'] = dev_ps
        f['dev_Qs'] = dev_Qs
        
        f['title_train_Xq'] = title_train_Xq
        f['title_train_Xp'] = title_train_Xp
        f['title_train_y'] = title_train_y
        f['title_dev_Xq'] = title_dev_Xq
        f['title_dev_Xp'] = title_dev_Xp
        f['title_dev_y'] = title_dev_y
        
        f['body_train_Xq'] = body_train_Xq
        f['body_train_Xp'] = body_train_Xp
        f['body_train_y'] = body_train_y
        f['body_dev_Xq'] = body_dev_Xq
        f['body_dev_Xp'] = body_dev_Xp
        f['body_dev_y'] = body_dev_y

        
        

    
    
