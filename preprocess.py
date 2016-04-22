import numpy as np
import h5py

CORPUS = 'data/askubuntu/text_tokenized.txt'
EMBEDDINGS = 'data/askubuntu/vector/vectors_pruned.200.txt'
TRAIN = 'data/askubuntu/train_random.txt'

def load_corpus(filename, word_dict):
    corpus = [[0 for _ in range(38)] for _ in range(523751)]
    with open(filename) as f:
        for line in f:
            line = line.split('\t')
            index, title = int(line[0]), map(lambda x : get_index(word_dict, x),
                                             filter(lambda x : len(x) > 0,
                                                    line[1][:-1].split(' ')))
            corpus[index] = title
            
    max_length = max(map(len, corpus))
    corpus = map(lambda x : x + [len(word_dict)] * (max_length - len(x)), corpus)
    
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
            
    word_dict["''"] = len(word_dict) + 1
    word_dict["/"] = len(word_dict) + 1
    word_dict[","] = len(word_dict) + 1
    word_dict["'s"] = len(word_dict) + 1
    word_dict["'m"] = len(word_dict) + 1
    word_dict["UNK"] = len(word_dict) + 1
    word_dict['END'] = len(word_dict) + 1
    
    embeddings.append(np.random.randn(200))
    embeddings.append(np.random.randn(200))
    embeddings.append(np.random.randn(200))
    embeddings.append(np.random.randn(200))
    embeddings.append(np.random.randn(200))
    embeddings.append(np.random.randn(200))
    embeddings.append(np.zeros(200))
    
    return word_dict, np.array(embeddings, dtype=np.float32)

def load_training(filename):
    qs = []
    ps = []
    Qs = []
    with open(filename, 'r') as f:
        for line in f:
            q, p, Q = line.split('\t')
            
            q = [int(q)]
            p = map(int, p.split(' '))

            Q = map(int, Q.split(' '))

            qs.append(q)
            ps.append(p)
            Qs.append(Q)
            
    max_length = max(map(len, ps))
    ps = map(lambda x : x + [0] * (max_length - len(x)), ps)
    
    return np.array(qs, np.int64), np.array(ps, np.int64), np.array(Qs, np.int64)
            
def get_index(word_dict, word):
    if word in word_dict:
        return word_dict[word]
    else:
        return word_dict['UNK']

if __name__ == '__main__':
    word_dict, embeddings = load_words(EMBEDDINGS)
    corpus = load_corpus(CORPUS, word_dict)
    qs, ps, Qs = load_training(TRAIN)
    print len(embeddings)

    with h5py.File('data/data.hdf5', 'w') as f:
        f['embeddings'] = embeddings
        f['corpus'] = corpus
        f['qs'] = qs
        f['ps'] = ps
        f['Qs'] = Qs
        

    
    
