import sys
sys.path.extend(['../qe'])

ReQue = {
    'parallel': 1,
    'op': ['generate', 'search', 'evaluate', 'build'],#['generate', 'search', 'evaluate', 'build']
    'expanders': {
        'SenseDisambiguation'   : 0,
        'Thesaurus'             : 1,
        'Wordnet'               : 0,
        'Conceptnet'            : 0,
        'Tagmee'                : 0,

        'Word2Vec'              : 0,
        'Glove'                 : 0,
        'Anchor'                : 0,
        'Wiki'                  : 0,

        'KrovetzStemmer'        : 1,
        'LovinsStemmer'         : 0,
        'PaiceHuskStemmer'      : 0,
        'PorterStemmer'         : 0,
        'Porter2Stemmer'        : 0,
        'SRemovalStemmer'       : 0,
        'Trunc4Stemmer'         : 0,
        'Trunc5Stemmer'         : 0,

        'RelevanceFeedback'     : 0,
        'Docluster'             : 0,
        'Termluster'            : 0,
        'Conceptluster'         : 0,
        'OnFields'              : 0,#make sure that the index for 'extcorpus' is available
        'AdapOnFields'          : 0,#make sure that the index for 'extcorpus' is available
        'BertQE'                : 0,
        'RM3'                   : 0,
    }
}

anserini = {
    'path': '../anserini/'
}

corpora = {
    'robust04': {
        'index': '../ds/robust04/lucene-index.robust04.pos+docvectors+rawdocs',
        'size': 528155,
        'topics': '../ds/robust04/topics.robust04.txt',
        'prels': '', #this will be generated after a retrieval {bm25, qld}
        'w_t': 2.25,#OnFields
        'w_a': 1,#OnFields
        'tokens': 148000000,
        'qrels':'../ds/robust04/qrels.robust04.txt',
        'extcorpus': 'gov2',#AdaptOnFields
    },
    'gov2': {
        'index': '../ds/gov2/lucene-index.gov2.pos+docvectors+rawdocs',
        'size': 25000000,
        'topics': '../ds/gov2/{}.terabyte0{}.txt',#{} is a placeholder for subtopics in main.py -> run()
        'prels': '',#this will be generated after a retrieval {bm25, qld}
        'w_t': 4,#OnFields
        'w_a': 0.25,#OnFields
        'tokens': 17000000000,
        'qrels':'../ds/gov2/qrels.terabyte0{}.txt',#{} is a placeholder for subtopics in main.py -> run()
        'extcorpus': 'robust04',#AdaptOnFields
    },
    'clueweb09b': {
        'index': '../ds/clueweb09b/lucene-index.cw09b.pos+docvectors+rawdocs',
        'size': 50000000,
        'topics': '../ds/clueweb09b/topics.web.{}.txt',#{} is a placeholder for subtopics in main.py -> run()
        'prels': '',#this will be generated after a retrieval {bm25, qld}
        'w_t': 1,#OnFields
        'w_a': 0,#OnFields
        'tokens': 31000000000,
        'qrels':'../ds/clueweb09b/qrels.web.{}.txt',#{} is a placeholder for subtopics in main.py -> run()
        'extcorpus': 'gov2',#AdaptOnFields
    },
    'clueweb12b13': {
        'index': '../ds/clueweb12b13/lucene-index.cw12b13.pos+docvectors+rawdocs',
        'size': 50000000,
        'topics': '../ds/clueweb12b13/topics.web.{}.txt',#{} is a placeholder for subtopics in main.py -> run()
        'prels': '',#this will be generated after a retrieval {bm25, qld}
        'w_t': 4,#OnFields
        'w_a': 0,#OnFields
        'tokens': 31000000000,
        'qrels':'../ds/clueweb12b13/qrels.web.{}.txt',#{} is a placeholder for subtopics in main.py -> run()
        'extcorpus': 'gov2',#AdaptOnFields
    },
        'antique': {
        'index': '../ds/antique/lucene-index-antique',
        'size': 403000,
        'topics': '../ds/antique/topics.antique.txt',
        'prels': '',#this will be generated after a retrieval {bm25, qld}
        'w_t': 2.25,#OnFields # to be tuned
        'w_a': 1,#OnFields # to be tuned
        'tokens': 16000000,
        'qrels':'../ds/antique/qrels.antique.txt',
        'extcorpus': 'gov2',#AdaptOnFields
    },
    'trec09mq': {
        'index': '../ds/trec09mq/lucene-index-trec09mq',
        'size': 50000000,
        'topics': '../ds/trec2009mq/prep/09.mq.topics.20001-60000.prep.tsv',
        'topics': '../ds/trec09mq/prep/09.mq.topics.20001-60000.prep',
        'prels': '',#this will be generated after a retrieval {bm25, qld}
        'w_t': 2.25,#OnFields # to be tuned
        'w_a': 1,#OnFields # to be tuned
        'tokens': 16000000,
        'qrels':'../ds/trec2009mq/prep/prel.20001-60000.prep.tsv',
        'qrels':'../ds/trec09mq/prep/prel.20001-60000.prep',
        'extcorpus': 'gov2',#AdaptOnFields
    },
        'dbpedia': {
        'index': '../ds/dbpedia/lucene-index-dbpedia-collection',
        'size': 4632359,
        'topics': '../ds/dbpedia/topics.dbpedia.txt',
        'prels': '',#this will be generated after a retrieval {bm25, qld}
        'w_t': 1,#OnFields # to be tuned
        'w_a': 1,#OnFields # to be tuned
        'tokens': 200000000,
        'qrels':'../ds/dbpedia/qrels.dbpedia.txt',
        'extcorpus': 'gov2',#AdaptOnFields
    }
}

