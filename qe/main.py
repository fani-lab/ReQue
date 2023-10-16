# TODO: list all library requirements such as stemmers, tagme, ...
import os, traceback, math, threading, time
import pandas as pd
import argparse

# Sparse Retrieval
from pyserini.search.lucene import LuceneSearcher, querybuilder
# Dense Retrieval
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder

# build anserini (maven) for doing A) indexing, B) information retrieval, and C) evaluation
# A) INDEX DOCUMENTS
# robust04
# $> ../anserini/target/appassembler/bin/IndexCollection -collection TrecCollection -input Robust04-Corpus -index lucene-index.robust04.pos+docvectors+rawdocs -generator JsoupGenerator -threads 44 -storePositions -storeDocvectors -storeRawDocs 2>&1 | tee log.robust04.pos+docvectors+rawdocs &
# Already done in https://git.uwaterloo.ca/jimmylin/anserini-indexes/raw/master/index-robust04-20191213.tar.gz

# Gov2:
# $> ../anserini/target/appassembler/bin/IndexCollection -collection TrecwebCollection -input Gov2-Corpus -index lucene-index.gov2.pos+docvectors+rawdocs -generator JsoupGenerator -threads 44 -storePositions -storeDocvectors -storeRawDocs 2>&1 | tee log.gov2.pos+docvectors+rawdocs &

# ClueWeb09-B-Corpus:
# $> ../anserini/target/appassembler/bin/IndexCollection -collection ClueWeb09Collection -input ClueWeb09-B-Corpus -index lucene-index.cw09b.pos+docvectors+rawdocs -generator JsoupGenerator -threads 44 -storePositions -storeDocvectors -storeRawDocs 2>&1 | tee  log.cw09b.pos+docvectors+rawdocs &

# ClueWeb12-B-Corpus:
# $> ../anserini/target/appassembler/bin/IndexCollection -collection ClueWeb12Collection -input ClueWeb12-B-Corpus -index lucene-index.cw12b13.pos+docvectors+rawdocs -generator JsoupGenerator -threads 44 -storePositions -storeDocvectors -storeRawDocs 2>&1 | tee  log.cw12b13.pos+docvectors+rawdocs &


# B) INFORMATION RETREIVAL: Ranking & Reranking
# $> ../anserini/target/appassembler/bin/SearchCollection -bm25 -threads 44 -topicreader Trec -index ../ds/robust04/index-robust04-20191213 -topics ../ds/robust04/topics.robust04.txt -output ./output/robust04/topics.robust04.bm25.txt

# C) EVAL
# $> ../anserini/eval/trec_eval.9.0.4/trec_eval -q -m map ../ds/robust04/qrels.robust04.txt ./output/robust04/topics.robust04.bm25.map.txt


# q: query
# Q: set of queries
# q_: expanded query (q')
# Q_: set of expanded queries(Q')
from cmn import param, utils
from cmn import expander_factory as ef
from expanders.abstractqexpander import AbstractQExpander
from expanders.onfields import OnFields


# from expanders.bertqe import BertQE

def generate(Qfilename, expander, output):
    model_name = expander.get_model_name()
    try:
        Q_filename = f'{output}.{model_name}.txt'
        expander.generate_queue(Qfilename, Q_filename)
    except: print(f'INFO: MAIN: GENERATE: There has been error in {expander}!\n{traceback.format_exc()}'); raise



def search(expander, rankers, topicreader, corpus, hitsnumber, output):
    model_name = expander.get_model_name()
    try:
        Q_filename = f'{output}.{model_name}.txt'
        if expander.query_set.empty: expander.query_set = expander.read_queries(Q_filename)
        for ranker in rankers:
            Q_pred = f'{output}.{model_name}.{utils.get_ranker_name(ranker)}.txt'
            q_dic = {}

            index = param.corpora[corpus]['dense_index'] if ranker == '-tct_colbert' else param.corpora[corpus]['index']

            if ranker == '-tct_colbert':
                encoder = TctColBertQueryEncoder(param.settings['encoder'])
                searcher = FaissSearcher(index, encoder)
            else:
                searcher = LuceneSearcher(index)
                if ranker == '-bm25': searcher.set_bm25(0.9, 0.4)
                elif ranker == '-qld': searcher.set_qld()

            if isinstance(expander, OnFields):  # or isinstance(expander, BertQE)
                run_file = open(Q_pred, 'w')
                list_of_raw_queries = utils.get_raw_query(topicreader, Q_filename)
                for qid, query in list_of_raw_queries.items(): q_dic[qid.strip()] = eval(query)
                for qid in q_dic.keys():
                    boost = []
                    for q_terms, q_weights in q_dic[qid].items():
                        try: boost.append(querybuilder.get_boost_query(querybuilder.get_term_query(q_terms), q_weights))
                        except: pass  # term do not exist in the indexed collection () e.g., stop words

                    should = querybuilder.JBooleanClauseOccur['should'].value
                    boolean_query_builder = querybuilder.get_boolean_query_builder()
                    for boost_i in boost: boolean_query_builder.add(boost_i, should)
                    retrieved_docs = []
                    query = boolean_query_builder.build()
                    hits = searcher.search(query, k=hitsnumber)
                    for i in range(0, hitsnumber):
                        try:
                            if hits[i].docid not in retrieved_docs:
                                retrieved_docs.append(hits[i].docid)
                                run_file.write(f'{qid} Q0  {hits[i].docid:15} {i + 1:2}  {hits[i].score:.5f} Pyserini \n')
                        except: pass
                run_file.close()

            else:
                with open(Q_pred, 'w', encoding='UTF-8') as run_file:
                    for index, row in expander.query_set.iterrows():
                        retrieved_docs = []
                        qid, qtext = row['qid'], row[model_name]
                        hits = searcher.search(qtext, k=hitsnumber)
                        for i in range(len(hits)):
                            if hits[i].docid not in retrieved_docs:
                                retrieved_docs.append(hits[i].docid)
                                run_file.write(f'{qid} Q0  {hits[i].docid:15} {i + 1:2} {hits[i].score:.5f} Pyserini\n')

    # all exception related to calling the SearchCollection cannot be captured here!! since it is outside the process scope
    except: print(f'INFO: MAIN: SEARCH: There has been error in {expander}!\n{traceback.format_exc()}'); raise


def evaluate(expander, Qrels, rankers, metrics, output):
    # Evaluation using trec_eval
    model_name = expander.get_model_name()
    try:
        for ranker in rankers:
            Q_pred = f'{output}.{model_name}.{utils.get_ranker_name(ranker)}.txt'
            for metric in metrics:
                Q_eval = f'{output}.{model_name}.{utils.get_ranker_name(ranker)}.{metric}.txt'
                cli_cmd = f'"{param.settings["treclib"]}" -m {metric} -q {Qrels} {Q_pred} > {Q_eval}'
                print(cli_cmd)
                stream = os.popen(cli_cmd)
                print(stream.read())
    # all exception related to calling the trec_eval cannot be captured here!! since it is outside the process scope
    except: print(f'INFO: MAIN: EVALUATE: There has been error in {expander}!\n{traceback.format_exc()}')


def aggregate(expanders, rankers, metrics, output):
    df = pd.DataFrame()
    # model_errs = dict()
    # queryids = pd.DataFrame()
    for model in expanders:
        model_name = model.get_model_name()
        # try:
        Q_filename = f'{output}.{model_name}.txt'
        # Q_ = model.read_expanded_queries(Q_filename)
        if model.query_set.empty: model.query_set = model.read_queries(Q_filename)
        for ranker in rankers:
            for metric in metrics:
                Q_eval = f'{output}.{model_name}.{utils.get_ranker_name(ranker)}.{metric}.txt'
                # the last row is average over all. skipped by [:-1]
                values = pd.read_csv(Q_eval, usecols=[1, 2], names=['qid', 'value'], header=None, sep='\t')[:-1]
                values.set_index('qid', inplace=True, verify_integrity=True)
                for idx, r in model.query_set.iterrows(): model.query_set.loc[idx, f'{model_name}.{utils.get_ranker_name(ranker)}.{metric}'] = values.loc[str(r.qid), 'value'] if str(r.qid) in values.index else None
        # except:
        #     model_errs[model_name] = traceback.format_exc()
        #     continue
        df = pd.concat([df, model.query_set], axis=1)

    filename = f"{output}.{'.'.join([utils.get_ranker_name(r) for r in rankers])}.{'.'.join(metrics)}.all.csv"
    df.to_csv(filename, index=False)
    # for model_err, msg in model_errs.items():
    #     print(f'INFO: MAIN: AGGREGATE: There has been error in {model_err}!\n{msg}')
    return filename


def build(input, expanders, rankers, metrics, output):
    base_model_name = AbstractQExpander().get_model_name()
    df = pd.read_csv(input, encoding='UTF-8')
    ds_df = df.iloc[:, :1 + 1 + len(rankers) * len(metrics)]  # the original query info
    ds_df['star_model_count'] = 0
    for idx, row in df.iterrows():
        star_models = dict()
        for model in expanders:
            model_name = model.get_model_name()
            if model_name == base_model_name: continue
            flag, sum = True, 0
            for ranker in rankers:
                for metric in metrics:
                    v = df.loc[idx, f'{model_name}.{utils.get_ranker_name(ranker)}.{metric}']
                    v = v if not pd.isna(v) else 0
                    v0 = df.loc[idx, f'{base_model_name}.{utils.get_ranker_name(ranker)}.{metric}']
                    v0 = v0 if not pd.isna(v0) else 0
                    if v <= v0: flag = False; break
                    sum += v ** 2
            if flag: star_models[model] = sum

        if len(star_models) > 0:
            ds_df.loc[idx, 'star_model_count'] = len(star_models.keys())
            star_models_sorted = {k: v for k, v in sorted(star_models.items(), key=lambda item: item[1], reverse=True)}
            for i, star_model in enumerate(star_models_sorted.keys()):
                ds_df.loc[idx, f'method.{i + 1}'] = star_model.get_model_name()
                ds_df.loc[idx, f'metric.{i + 1}'] = math.sqrt(star_models[star_model])
                ds_df.loc[idx, f'query.{i + 1}'] = df.loc[idx, f'{star_model.get_model_name()}']
        else: ds_df.loc[idx, 'star_model_count'] = 0
    filename = f"{output}.{'.'.join([utils.get_ranker_name(r) for r in rankers])}.{'.'.join(metrics)}.dataset.csv"
    ds_df.to_csv(filename, index=False, encoding='UTF-8')
    return filename


def worker(corpus, rankers, metrics, op, output_, topicreader, expanders):
    exceptions = {}

    #TODO: make it message queue
    def worker_thread(expander):
        try:
            if 'generate' in op: generate(Qfilename=param.corpora[corpus]['topics'], expander=expander, output=output_)
            if 'search' in op: search(expander=expander, rankers=rankers, hitsnumber=param.settings['hitsnumber'], topicreader=topicreader, corpus=corpus, output=output_)
            if 'evaluate' in op: evaluate(expander=expander, Qrels=param.corpora[corpus]['qrels'], rankers=rankers, metrics=metrics, output=output_)
        except:
            print(f'INFO: MAIN: THREAD: {threading.currentThread().getName()}: There has been error in {expander}!\n{traceback.format_exc()}')
            exceptions[expander.get_model_name()] = traceback.format_exc()

    threads = []
    for expander in expanders:
        if param.ReQue['parallel']: threads.append(threading.Thread(daemon=True, target=worker_thread, name=expander.get_model_name(), args=(expander,)))
        else: worker_thread(expander)
    if param.ReQue['parallel']: print(f'Starting threads per expanders for {[e for e in param.ReQue["op"] if e != "build"]} ...')
    for thread in threads: thread.start()
    return threads, exceptions


def initialize(corpus, rankers, metrics, output, rf=True, op=[], topicreader=""):
    expanders = ef.get_nrf_expanders()
    # local analysis
    if rf: expanders += ef.get_rf_expanders(rankers=rankers, corpus=corpus, output=output, ext_corpus=param.corpora[corpus]['extcorpus'])

    threads, exceptions = worker(corpus=corpus, rankers=rankers, metrics=metrics, op=op, output_=output, topicreader=topicreader, expanders=expanders)
    for thread in threads: thread.join()
    expanders = [e for e in expanders if e.get_model_name() not in exceptions.keys()]

    if 'build' in op:
        result = aggregate(expanders=expanders, rankers=rankers, metrics=metrics, output=output)
        build(input=result, expanders=expanders, rankers=rankers, metrics=metrics, output=output)
    else: result = None
    return result


def run(corpus, rankers, metrics, output, rf=True, op=[]):
    r = []

    if corpus == 'dbpedia': topicreader = 'TsvString'
    elif corpus == 'antique': topicreader = 'TsvInt'
    elif corpus == 'trec09mq': topicreader = 'TsvInt'
    elif corpus == 'orcas': topicreader = 'TsvInt'
    # The 672 query (topic) has no qrels (document judge relevant)
    elif corpus == 'robust04': topicreader = 'Trec'
    elif corpus == 'gov2':  topicreader = 'Trec'; output += 'topics.terabyte0';  r = ['4.701-750', '5.751-800', '6.801-850']; number = '701-850'; results = []
    elif corpus == 'clueweb09b': topicreader = 'Webxml'; output += 'topics.web.'; r = ['1-50', '51-100', '101-150', '151-200']; number = '1-200'; results = []
    elif corpus == 'clueweb12b13': topicreader = 'Webxml'; output += 'topics.web.'; r = ['201-250', '251-300']; number = '201-300'; results = []
    else: print('Please choose a corpus between these:' + ', '.join(param.corpora.keys())); exit()

    if len(r) == 0:
        output_ = f'{output}topics.{corpus}'
        result = initialize(corpus, rankers, metrics, output_, rf, op, topicreader)

    else:
        topics = param.corpora[corpus]['topics']
        qrels = param.corpora[corpus]['qrels']
        for i in r:
            output_ = output + i
            param.corpora[corpus]['topics'] = topics.replace("{}", i)
            param.corpora[corpus]['qrels'] = qrels.replace("{}", i)
            result = initialize(corpus, rankers, metrics, output_, rf, op, topicreader)
            if 'build' in op: results.append(result)

        if 'build' in op:
            output_ = results[0].replace('.' + results[0].split('/')[-1].split('.')[1] + '.', f'.{corpus}.').replace(results[0].split('/')[-1].split('.')[2], number)
            df = pd.DataFrame()
            for r in results: df = pd.concat([df, pd.read_csv(r)], axis=0, ignore_index=True, sort=False)
            df.to_csv(output_, index=False)


def addargs(parser):
    corpus = parser.add_argument_group('Corpus')
    corpus.add_argument('--corpus', type=str, choices=['dbpedia', 'antique', 'robust04', 'gov2', 'clueweb09b', 'clueweb12b13', 'trec09mq', 'orcas', 'testds'], required=True, help='The corpus name; required; (example: robust04)')
    gold = parser.add_argument_group('Gold Standard Dataset')
    gold.add_argument('--output', type=str, required=True, help='The output path for the gold standard dataset; required; (example: ./output/robust04/')
    gold.add_argument('--rankers', nargs='+', type=str.lower, choices=['bm25', 'qld', 'tct_colbert'], default=['bm25'], help='The ranker names (default: bm25 qld)')
    gold.add_argument('--metrics', nargs='+', type=str.lower, choices=['map', 'ndcg', 'recip_rank'], default=['map'], help='The evaluation metric names (default: map ndcg)')


# # python -u main.py --corpus robust04 --output ./output/robust04/ --rankers bm25 qld --metrics map ndcg 2>&1 | tee robust04.log &
# # python -u main.py --corpus gov2 --output ./output/gov2/ --ranker bm25 qld --metrics map ndcg 2>&1 | tee gov2.bm25.log &
# # python -u main.py --corpus clueweb09b --output ./output/clueweb09b/ --ranker bm25 qld --metrics map ndcg 2>&1 | tee clueweb09b.log &
# # python -u main.py --corpus clueweb12b13 --output ./output/clueweb12b13/ --ranker bm25 qld --metrics map ndcg 2>&1 | tee clueweb12b13.log &
# # python -u main.py --corpus antique --output ./output/antique/ --ranker bm25 qld --metrics map ndcg 2>&1 | tee antique.log &
# # python -u main.py --corpus trec09mq --output ./output/trec09mq/ --ranker bm25 qld --metrics map ndcg 2>&1 | tee trec09mq.log &
# # python -u main.py --corpus orcas --output ./output/orcas/ --ranker bm25 qld --metrics map ndcg 2>&1 | tee trec09mq.log &

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ReQue (Refining Queries)')
    addargs(parser)
    args = parser.parse_args()

    ## rf: whether to include relevance feedback expanders (local analysis) or not
    ## op: determines the steps in the pipeline. op=['generate', 'search', 'evaluate', 'build']

    run(corpus=args.corpus.lower(),
        rankers=['-' + ranker for ranker in args.rankers],
        metrics=args.metrics,
        output=args.output + args.corpus.lower() + '/',
        rf=True,
        op=param.ReQue['op'])
