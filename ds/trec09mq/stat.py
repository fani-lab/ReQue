import pandas as pd
import  pickle
import seaborn as sns
import matplotlib.pyplot as plt

def read_refinements(all, dataset, querytype):

    df_all=pd.read_csv(all, encoding="ISO-8859-1", sep=',')
    R = pd.read_csv(all, encoding="ISO-8859-1", sep=',').columns[4:len(df_all.columns):3]
    df_results= pd.read_csv(dataset, encoding="ISO-8859-1", sep=',')
    df_qt=pd.read_csv(querytype, encoding="ISO-8859-1", sep='\t',usecols =['Topic', 'Class'])
    df_qt['Topic']=df_qt['Topic'].astype(int)
    df_qt.rename(columns=({'Topic': 'qid'}), inplace=True)
    df = pd.merge(df_results, df_qt, on='qid', how='inner')
    QT = df_qt['Class'].unique()

    A=dict()
    for qt in QT:
        A[qt]=dict()
        for r in R: A[qt][r]=0

    for index,row in df.iterrows():
        r_n=row['star_model_count']*3
        for i in range(1,r_n,3):
            r=str(row[i+3])
            qt=row["Class"]
            A[qt][r] +=1

    with open('final_refinements.pkl', 'wb') as f:
        pickle.dump(A, f)
    return  R, QT

def heatmap(x, y):
    with open('final_refinements.pkl', 'rb') as f:
        refinements = pickle.load(f)
    df = pd.DataFrame.from_dict(refinements)
    df=df.transpose()
    x_axis_labels = x
    y_axis_labels =y
    sns.heatmap(df, xticklabels=x_axis_labels, yticklabels=y_axis_labels);
    plt.title("Improvement of the initial queries \n by different query expanders given a query type.", fontsize=12)
    plt.savefig("../../qe/output/trec09mq/stat/heatmap.png", bbox_inches='tight', dpi=100)
    plt.show()

if __name__ == "__main__":
    refinements, querytypes= read_refinements('../../qe/output/trec09mq/topics.trec09mq.bm25.map.all.csv','../../qe/output/trec09mq/topics.trec09mq.bm25.map.dataset.csv','./queryclasses')
    heatmap( refinements, querytypes)