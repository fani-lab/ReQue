import pandas as pd
import numpy as np
import sys
import pickle
from pandas.core.dtypes.missing import isna



#delet colmun and creat new file from original dataset
def edit_trec2009MQ(dataset):
    q = pd.read_csv(f'../../ds/trec2009mq/source/{dataset}', encoding="ISO-8859-1", sep=':', index_col=False)
    data=q.drop(q.columns[1],axis=1)
    data.to_csv(r'../../ds/trec2009mq/prep/prels.20001-60000.prep.tsv', header=None, index=None, sep='\t', mode='a')

# extract queris with query types and store in new txt file named 09.mq.topics.20001-60000.prep.tsv
def find_fquery(types):
    qt= pd.read_csv(f'../../ds/trec2009mq/source/{types}', encoding = "ISO-8859-1", sep='\t', index_col=False)
    q = pd.read_csv(f'prep/prels.20001-60000.prep.tsv', encoding ="ISO-8859-1", sep='\t', index_col=False, names=['id', 'query'])
    for i in range(len(q)):
        for j in range(len(qt)):
            if str(qt["Topic"][j]) == str(q["id"][i]):
                if str(qt["Class"][j])== "Information_Close":
                    (q.loc[[i]]).to_csv(r'../../ds/trec2009mq/prep/09.mq.topics.20001-60000.prep.tsv', header=None, index=None, sep='\t', mode='a')
                if str(qt["Class"][j]) == "Information_Open":
                    (q.loc[[i]]).to_csv(r'../../ds/trec2009mq/prep/09.mq.topics.20001-60000.prep.tsv', header=None, index=None, sep='\t', mode='a')
                if str(qt["Class"][j]) == "Navigational":
                    (q.loc[[i]]).to_csv(r'../../ds/trec2009mq/prep/09.mq.topics.20001-60000.prep.tsv', header=None, index=None, sep='\t', mode='a')
                if str(qt["Class"][j]) == "Resource":
                    (q.loc[[i]]).to_csv(r'../../ds/trec2009mq/prep/09.mq.topics.20001-60000.prep.tsv', header=None, index=None, sep='\t', mode='a')
                if str(qt["Class"][j]) == "Advice":
                    (q.loc[[i]]).to_csv(r'../../ds/trec2009mq/prep/09.mq.topics.20001-60000.prep.tsv', header=None, index=None, sep='\t', mode='a')

#09.mq.topics.20001-60000.prep.tsv
#prels.20001-60000.prep.tsv

if __name__ == "__main__":
    query = sys.argv[1]
    types = sys.argv[2]
    qrel = sys.argv[3]

    edit_trec2009MQ(query)
    find_fquery(types)

