from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
import numpy as np

def preprocess_data(dataset, qtypes):
    df_results= pd.read_csv(dataset, sep=',')
    df_results.drop(df_results[df_results['star_model_count'] ==0].index, inplace=True) # drop rows with star_model_count=0
    qtypes = pd.read_csv(qtypes, sep='\t', usecols=['Topic', 'Class'])
    qtypes['Topic'] = qtypes['Topic'].astype(int)
    qtypes.rename(columns=({'Topic': 'qid'}), inplace=True)
    df_results = pd.merge(df_results, qtypes, on='qid', how='inner')

    x, y = df_results["Class"], df_results["method.1"]
    ohe = OneHotEncoder()
    enc_x = ohe.fit_transform(np.asarray(x).reshape(-1,1))
    x=enc_x.toarray()
    enc_y = ohe.fit_transform(np.asarray(y).reshape(-1, 1))
    y = enc_y.toarray()

    x_train, x_test, y_train,y_test = train_test_split(x, y, train_size=0.75)
    print(x_train.shape)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    rf.fit(x_train, y_train)
    test=rf.score(x_train,y_train)
    print(test)
   # predict=rf.predict(x_train)



if __name__ == "__main__":
    preprocess_data('../../qe/output/trec09mq/topics.trec09mq.bm25.map.dataset.csv',
                    './queryclasses')

