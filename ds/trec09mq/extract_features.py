from scipy import sparse
import pandas as pd
import numpy as np
import os

from lib import text_corpus as tc, utils

def extract_features(Q, feature_set=[]):#['basic', 'linguistic', 'w2v', 'bert']
    char_ngram_range = (1, 1)
    word_ngram_range = (1, 1)
    tc_Q = tc.TextCorpus(Q, word_pattern=r"[0-9]+", char_ngram_range=char_ngram_range, word_ngram_range=word_ngram_range)

    q_features = sparse.csr_matrix(tc_Q.getLengthsByTerm()).transpose()
    print('tc_Q',tc_Q)

    if 'basic' in feature_set: q_features = sparse.csr_matrix(sparse.hstack((
        q_features,
        #tc_Q.getCharStat()[0], yandex words are ids
        tc_Q.getTermStat()[0],
        tc_Q.getTfIdF(),
        # tc_Q.getCharBM25(),
        # tc_Q.getTermBM25()
    )))
    #print("getTermStat", tc_Q.getTermStat()[0])
    print("getTfIdF", tc_Q.getTfIdF())

    return q_features

def extract_load_q_features(Q, y, feature_set, q_features_file=None):
    print("kkkkkkkk",y)
    try:
        features_label = utils.load_sparse_csr(q_features_file)
        x = features_label[:, :-1]
        y = features_label[:, -1:]
        print("lllllllllllllllll",x)
    except:
        x = extract_features(Q, feature_set)
        print(x.shape, y.shape)
        # print(x)
        # print(y.values.reshape(y.shape[0],1))
        x_y = sparse.csr_matrix(sparse.hstack((x, y.values.reshape(y.shape[0],1))))
        #rint("llpppppppppppp", x_y)
        #utils.save_sparse_csr(q_features_file, x_y)




    # try:
    #     features_label = utils.load_sparse_csr(q_features_file)
    #     return features_label[:, :-1], features_label[:, -1:]
    # except:
    #     q_features = extract_features(Q, feature_set)
    #     q_features_label = sparse.csr_matrix(sparse.hstack((q_features, y)))
    #     utils.save_sparse_csr(q_features_file, q_features_label)
    #     print(f"saved query features with shape (data size, feature size + topn): {q_features.shape}")
    #     return q_features_label[:, :-1], q_features[:, -1:]