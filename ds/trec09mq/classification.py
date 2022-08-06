from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import dal
import extract_features as ef

def preprocess_data(dataset, qtypes):
    df_results= pd.read_csv(dataset, sep=',')
    #df_results.drop(df_results[df_results['star_model_count'] ==0].index, inplace=True) # drop rows with star_model_count=0
    qtypes = pd.read_csv(qtypes, sep='\t', usecols=['Topic', 'Class'])
    qtypes['Topic'] = qtypes['Topic'].astype(int)
    qtypes.rename(columns=({'Topic': 'qid'}), inplace=True)
    df = pd.merge(df_results, qtypes, on='qid', how='inner')
    return  df

def create_evaluation_splits(n_sample, n_folds):
    train, test = train_test_split(np.arange(n_sample), train_size=0.7, test_size=0.3, random_state=0, shuffle=True)
    splits = dict()
    splits['test'] = test
    splits['folds'] = dict()
    skf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
    for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
        splits['folds'][k] = dict()
        splits['folds'][k]['train'] = train[trainIdx]
        #splits['folds'][k]['valid'] = train[validIdx]

    return splits

def main(splits, Q, y,feature_set, path, cmd=['prep', 'train', 'eval', 'test']):
    q_features, q_labels = ef.extract_load_q_features(Q, y,feature_set, q_features_file=path)
    #print(q_features)
#     if 'userid' in feature_set:
#         userID_col = np.array(Q['UserID'])[:,None]
#         q_features = sparse.hstack([userID_col, q_features]).tocsr()
#
#     cores = multiprocessing.cpu_count()
#     models = [LogisticRegression(n_jobs=cores, max_iter=100, random_state=0),
#               RandomForestClassifier(n_jobs=cores, n_estimators=100, max_depth=10, random_state=0),  #
#               # XGBClassifier(n_jobs=cores, n_estimators=1000, max_depth=15, learning_rate=0.1, random_state=random_seed),
#               # MLP (word embdedding in input, n_refiners in output)
#               # multilabel
#               ]
#
#     output = f'{path}.qf'
#
#     for model in models:
#         if 'train' in cmd: learn_kf(q_features, q_labels.toarray().ravel(), model, splits=splits, Q=Q, output=output)
#         if 'test' in cmd: test(q_features, q_labels.toarray().ravel(), model, splits=splits, Q=Q, output=output)


def con_r2i(df):
    r2i = {}
    r= df['method.1'].unique()
    #print(r)
    for i in range(len(r)):
        r2i[r[i]]=i
    #print(r2i)
    for i in range(len(df)):
        # replace refinement with index
        df['method.1'][i]=r2i[df['method.1'][i]]
        #x=r2i[df['method.1'][i]]
        #df['method.1'] = df['method.1'].replace([df['method.1'][i]], int(x))
       # df.set_value('method.1', i, r2i[df['method.1'][i]])
        #f.loc[df['method.1'] == 'n', col] = 0
        #print(col)
        #x=r2i[df['method.1'][i]]
        #df.loc[:, 'method.1'][i]=
    #print(df['method.1'])
    return df

def con_t2i(df):
    t2i = {}
    t= df['Class'].unique()
    print(t)
    for i in range(len(t)):
        t2i[t[i]]=i
    print(t2i)
    for i in range(len(df)):
        # replace refinement with index
        df['Class'][i]=t2i[df['Class'][i]]
    print(df['Class'])
    # the df contain both refinemnets and qtypes indexes
    return df




if __name__ == "__main__":
    df_results=preprocess_data('../../qe/output/trec09mq/topics.trec09mq.bm25.map.dataset.csv',
                    './queryclasses')
    splits = create_evaluation_splits(len(df_results), 5)
    feature_sets = [['basic'], ['userid', 'basic']]
    con_r2i(df_results)
    # the df contain both refinemnets and qtypes indexes
    df=con_t2i(con_r2i(df_results))
    feature_sets = [['basic'], ['userid', 'basic']]
    # for feature_set in feature_sets:
    #     feature_set_str = '.'.join(feature_set)
    #     main(splits, df['abstractqueryexpansion'], df['method.1'],feature_set, f'./results')



    # try:
    #     features_label = utils.load_sparse_csr(q_features_file)
    #     x=features_label[:, :-1]
    #     y=features_label[:, -1:]
    # except:
    #     x = extract_features(df_results['abstractqueryexpansion'], feature_set)
    #     x_y = sparse.csr_matrix(sparse.hstack((x, df_results["method.1"])))
    #     utils.save_sparse_csr(q_features_file, q_features_label)


    # if x = x + qtype

    #print(r2i_x,'\n', i2r_x)
   ##feature_set = ['basic']
    #main(splits, Q, r2i_x,feature_set, f'./results')
    # for feature_set in feature_sets:
    #     feature_set_str = '.'.join(feature_set)
    #     main(splits, Q, feature_set, f'../../output/yandex/yandex.{feature_set_str}')


