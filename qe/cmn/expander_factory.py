import sys
sys.path.extend(['../qe'])
sys.path.extend(['../qe/cmn'])

from expanders.abstractqexpander import AbstractQExpander
from expanders.stem import Stem # Stem expander is the wrapper for all stemmers as an expnader :)

import param
from cmn import utils

#global analysis
def get_nrf_expanders():
    expanders_list = [AbstractQExpander()]
    if param.ReQue['expanders']['Thesaurus']: from expanders.thesaurus import Thesaurus; expanders_list.append(Thesaurus())
    if param.ReQue['expanders']['Thesaurus']: from expanders.thesaurus import Thesaurus; expanders_list.append(Thesaurus(replace=True))
    if param.ReQue['expanders']['Wordnet']: from expanders.wordnet import Wordnet; expanders_list.append(Wordnet())
    if param.ReQue['expanders']['Wordnet']: from expanders.wordnet import Wordnet; expanders_list.append(Wordnet(replace=True))
    if param.ReQue['expanders']['Word2Vec']: from expanders.word2vec import Word2Vec; expanders_list.append(Word2Vec('../pre/wiki-news-300d-1M.vec'))
    if param.ReQue['expanders']['Word2Vec']: from expanders.word2vec import Word2Vec; expanders_list.append(Word2Vec('../pre/wiki-news-300d-1M.vec', replace=True))
    if param.ReQue['expanders']['Glove']: from expanders.glove import Glove; expanders_list.append(Glove('../pre/glove.6B.300d'))
    if param.ReQue['expanders']['Glove']: from expanders.glove import Glove; expanders_list.append(Glove('../pre/glove.6B.300d', replace=True))
    if param.ReQue['expanders']['Anchor']: from expanders.anchor import Anchor; expanders_list.append(Anchor(anchorfile='../pre/anchor_text_en.ttl', vectorfile='../pre/wiki-anchor-text-en-ttl-300d.vec'))
    if param.ReQue['expanders']['Anchor']: from expanders.anchor import Anchor; expanders_list.append(Anchor(anchorfile='../pre/anchor_text_en.ttl', vectorfile='../pre/wiki-anchor-text-en-ttl-300d.vec', replace=True))
    if param.ReQue['expanders']['Wiki']: from expanders.wiki import Wiki; expanders_list.append(Wiki('../pre/temp_model_Wiki'))
    if param.ReQue['expanders']['Wiki']: from expanders.wiki import Wiki; expanders_list.append(Wiki('../pre/temp_model_Wiki', replace=True))
    if param.ReQue['expanders']['Tagmee']: from expanders.tagmee import Tagmee; expanders_list.append(Tagmee())
    if param.ReQue['expanders']['Tagmee']: from expanders.tagmee import Tagmee; expanders_list.append(Tagmee(replace=True))
    if param.ReQue['expanders']['SenseDisambiguation']: from expanders.sensedisambiguation import SenseDisambiguation; expanders_list.append(SenseDisambiguation())
    if param.ReQue['expanders']['SenseDisambiguation']: from expanders.sensedisambiguation import SenseDisambiguation; expanders_list.append(SenseDisambiguation(replace=True))
    if param.ReQue['expanders']['Conceptnet']: from expanders.conceptnet import Conceptnet; expanders_list.append(Conceptnet())
    if param.ReQue['expanders']['Conceptnet']: from expanders.conceptnet import Conceptnet; expanders_list.append(Conceptnet(replace=True))
    if param.ReQue['expanders']['KrovetzStemmer']: from stemmers.krovetz import KrovetzStemmer; expanders_list.append(Stem(KrovetzStemmer(jarfile='stemmers/kstem-3.4.jar')))
    if param.ReQue['expanders']['LovinsStemmer']: from stemmers.lovins import LovinsStemmer; expanders_list.append(Stem(LovinsStemmer()))
    if param.ReQue['expanders']['PaiceHuskStemmer']: from stemmers.paicehusk import PaiceHuskStemmer; expanders_list.append(Stem(PaiceHuskStemmer()))
    if param.ReQue['expanders']['PorterStemmer']: from stemmers.porter import PorterStemmer; expanders_list.append(Stem(PorterStemmer()))
    if param.ReQue['expanders']['Porter2Stemmer']: from stemmers.porter2 import Porter2Stemmer; expanders_list.append(Stem(Porter2Stemmer()))
    if param.ReQue['expanders']['SRemovalStemmer']: from stemmers.sstemmer import SRemovalStemmer; expanders_list.append(Stem(SRemovalStemmer()))
    if param.ReQue['expanders']['Trunc4Stemmer']: from stemmers.trunc4 import Trunc4Stemmer; expanders_list.append(Stem(Trunc4Stemmer()))
    if param.ReQue['expanders']['Trunc5Stemmer']: from stemmers.trunc5 import Trunc5Stemmer; expanders_list.append(Stem(Trunc5Stemmer()))
    if param.ReQue['expanders']['BackTranslation']: from expanders.backtranslation import BackTranslation; expanders_list.extend([BackTranslation(each_lng) for index, each_lng in enumerate(param.backtranslation['tgt_lng'])])
    # since RF needs index and search output which depends on ir method and topics corpora, we cannot add this here. Instead, we run it individually
    # RF assumes that there exist abstractqueryexpansion files

    return expanders_list

#local analysis
def get_rf_expanders(rankers, corpus, output, ext_corpus=None):
    expanders_list = []
    for ranker in rankers:
        ranker_name = utils.get_ranker_name(ranker)
        if param.ReQue['expanders']['RM3']: from expanders.rm3 import RM3; expanders_list.append(RM3(ranker=ranker_name, index=param.corpora[corpus]['index']))
        if param.ReQue['expanders']['RelevanceFeedback']: from expanders.relevancefeedback import RelevanceFeedback; expanders_list.append(RelevanceFeedback(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), anserini=param.anserini['path'], index=param.corpora[corpus]['index']))
        if param.ReQue['expanders']['Docluster']: from expanders.docluster import Docluster; expanders_list.append(Docluster(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), anserini=param.anserini['path'], index=param.corpora[corpus]['index'])),
        if param.ReQue['expanders']['Termluster']: from expanders.termluster import Termluster; expanders_list.append(Termluster(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), anserini=param.anserini['path'], index=param.corpora[corpus]['index']))
        if param.ReQue['expanders']['Conceptluster']: from expanders.conceptluster import Conceptluster; expanders_list.append(Conceptluster(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), anserini=param.anserini['path'], index=param.corpora[corpus]['index']))
        if param.ReQue['expanders']['BertQE']: from expanders.bertqe import BertQE; expanders_list.append(BertQE(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), index=param.corpora[corpus]['index'], anserini=param.anserini['path']))
        if param.ReQue['expanders']['OnFields']: from expanders.onfields import OnFields; expanders_list.append(OnFields(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), anserini=param.anserini['path'], index=param.corpora[corpus]['index'], w_t=param.corpora[corpus]['w_t'], w_a=param.corpora[corpus]['w_a'], corpus_size=param.corpora[corpus]['size']))
        if param.ReQue['expanders']['AdapOnFields']: from expanders.adaponfields import AdapOnFields; expanders_list.append(AdapOnFields(ranker=ranker_name,prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), anserini=param.anserini['path'], index=param.corpora[corpus]['index'], w_t=param.corpora[corpus]['w_t'], w_a=param.corpora[corpus]['w_a'], corpus_size=param.corpora[corpus]['size'], collection_tokens=param.corpora[corpus]['tokens'], ext_corpus=ext_corpus, ext_index=param.corpora[ext_corpus]['index'], ext_collection_tokens=param.corpora[ext_corpus]['tokens'], ext_w_t=param.corpora[ext_corpus]['w_t'], ext_w_a=param.corpora[ext_corpus]['w_a'], ext_corpus_size=param.corpora[ext_corpus]['size'], adap=True))

    return expanders_list
