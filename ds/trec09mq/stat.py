import numpy as np
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt

# read data from final Reque prep (...map.dataset.csv) and build a matrix to draw the heatmap as a stat
def bulding_matrix():
    result = pd.read_csv("../../qe/output/trec2009mq/topics.antique.bm25.map.dataset.csv")
    classes = pd.read_csv(f'source/queryclasses', encoding='utf-8', sep='\t')
    row = 5
    col = 16

    matrix = [[0 for x in range(col)] for y in range(row)]
    for q in range(len(result)):
        for c in range(len(classes)):
            if str(result['qid'][q]) == str(classes["Topic"][c]):
                # row 0 is Information_Close type
                j = result["star_model_count"][q]
                if str(classes["Class"][c]) == "Information_Close":
                        for k in range(j,0,-1):
                             if k > 0:
                                 if str(result["method.{}".format(k)][q]) == "sensedisambiguation.replace":
                                     matrix[0][0] = matrix[0][0] + 1
                                 if str(result["method.{}".format(k)][q]) == "wordnet.topn3.replace":
                                     matrix[0][1] = matrix[0][1] + 1
                                 if str(result["method.{}".format(k)][q]) == "tagmee.topn3":
                                     matrix[0][2] = matrix[0][2] + 1
                                 if str(result["method.{}".format(k)][q]) == "word2vec.topn3.replace":
                                     matrix[0][3] = matrix[0][3] + 1
                                 if str(result["method.{}".format(k)][q]) in ("glove.topn3.replace" , "glove.topn3"):
                                     matrix[0][4] = matrix[0][4] + 1
                                 if str(result["method.{}".format(k)][q]) in ("wiki.topn3", "wiki.topn3.replace") :
                                     matrix[0][5] = matrix[0][5] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.krovetz":
                                     matrix[0][6] = matrix[0][6] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.porter":
                                     matrix[0][7] = matrix[0][7] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.porter2":
                                     matrix[0][8] = matrix[0][8] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.trunc4":
                                     matrix[0][9] = matrix[0][9] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.paicehusk":
                                     matrix[0][10] = matrix[0][10] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.trunc5":
                                     matrix[0][11] = matrix[0][11] + 1
                                 if str(result["method.{}".format(k)][q]) == "conceptluster.topn5.3.bm25":
                                     matrix[0][12] = matrix[0][12] + 1
                                 if str(result["method.{}".format(k)][q]) == "termluster.topn5.3.bm25":
                                     matrix[0][13] = matrix[0][13] + 1
                                 if str(result["method.{}".format(k)][q]) == "relevancefeedback.topn10.bm25":
                                     matrix[0][14] = matrix[0][14] + 1
                                 if str(result["method.{}".format(k)][q]) == "docluster.topn10.3.bm25":
                                     matrix[0][15] = matrix[0][15] + 1
                if str(classes["Class"][c]) == "Information_Open":
                         for k in range(j,0,-1):
                             if k > 0:
                                 if str(result["method.{}".format(k)][q]) == "sensedisambiguation.replace":
                                     matrix[1][0] = matrix[1][2] + 1
                                 if str(result["method.{}".format(k)][q]) == "wordnet.topn3.replace":
                                     matrix[1][1] = matrix[1][1] + 1
                                 if str(result["method.{}".format(k)][q]) == "tagmee.topn3":
                                     matrix[1][2] = matrix[1][2] + 1
                                 if str(result["method.{}".format(k)][q]) == "word2vec.topn3.replace":
                                     matrix[1][3] = matrix[1][3] + 1
                                 if str(result["method.{}".format(k)][q]) in ("glove.topn3.replace", "glove.topn3"):
                                     matrix[1][4] = matrix[1][4] + 1
                                 if str(result["method.{}".format(k)][q]) in ("wiki.topn3", "wiki.topn3.replace"):
                                     matrix[1][5] = matrix[1][5] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.krovetz":
                                     matrix[1][6] = matrix[1][6] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.porter":
                                     matrix[1][7] = matrix[1][7] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.porter2":
                                     matrix[1][8] = matrix[1][8] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.trunc4":
                                     matrix[1][9] = matrix[1][9] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.paicehusk":
                                     matrix[1][10] = matrix[1][10] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.trunc5":
                                     matrix[1][11] = matrix[1][11] + 1
                                 if str(result["method.{}".format(k)][q]) == "conceptluster.topn5.3.bm25":
                                     matrix[1][12] = matrix[1][12] + 1
                                 if str(result["method.{}".format(k)][q]) == "termluster.topn5.3.bm25":
                                     matrix[1][13] = matrix[1][13] + 1
                                 if str(result["method.{}".format(k)][q]) == "relevancefeedback.topn10.bm25":
                                     matrix[1][14] = matrix[1][14] + 1
                                 if str(result["method.{}".format(k)][q]) == "docluster.topn10.3.bm25":
                                     matrix[1][15] = matrix[1][15] + 1
                if str(classes["Class"][c]) == "Navigational":
                        for k in range(j,0,-1):
                            if k > 0:
                                if str(result["method.{}".format(k)][q]) == "sensedisambiguation.replace":
                                    matrix[2][0] = matrix[2][0] + 1
                                if str(result["method.{}".format(k)][q]) == "wordnet.topn3.replace":
                                    matrix[2][1] = matrix[2][1] + 1
                                if str(result["method.{}".format(k)][q]) == "tagmee.topn3":
                                    matrix[2][2] = matrix[2][2] + 1
                                if str(result["method.{}".format(k)][q]) == "word2vec.topn3.replace":
                                    matrix[2][3] = matrix[2][3] + 1
                                if str(result["method.{}".format(k)][q]) in ("glove.topn3.replace", "glove.topn3"):
                                    matrix[2][4] = matrix[2][4] + 1
                                if str(result["method.{}".format(k)][q]) in ("wiki.topn3", "wiki.topn3.replace"):
                                    matrix[2][5] = matrix[2][5] + 1
                                if str(result["method.{}".format(k)][q]) == "stem.krovetz":
                                    matrix[2][6] = matrix[2][6] + 1
                                if str(result["method.{}".format(k)][q]) == "stem.porter":
                                    matrix[2][7] = matrix[2][7] + 1
                                if str(result["method.{}".format(k)][q]) == "stem.porter2":
                                    matrix[2][8] = matrix[2][8] + 1
                                if str(result["method.{}".format(k)][q]) == "stem.trunc4":
                                    matrix[2][9] = matrix[2][9] + 1
                                if str(result["method.{}".format(k)][q]) == "stem.paicehusk":
                                    matrix[2][10] = matrix[2][10] + 1
                                if str(result["method.{}".format(k)][q]) == "stem.trunc5":
                                    matrix[2][11] = matrix[2][11] + 1
                                if str(result["method.{}".format(k)][q]) == "conceptluster.topn5.3.bm25":
                                    matrix[2][12] = matrix[2][12] + 1
                                if str(result["method.{}".format(k)][q]) == "termluster.topn5.3.bm25":
                                    matrix[2][13] = matrix[2][13] + 1
                                if str(result["method.{}".format(k)][q]) == "relevancefeedback.topn10.bm25":
                                    matrix[2][14] = matrix[2][14] + 1
                                if str(result["method.{}".format(k)][q]) == "docluster.topn10.3.bm25":
                                    matrix[2][15] = matrix[2][15] + 1
                if str(classes["Class"][c]) == "Resource":
                         for k in range(j,0,-1):

                             if k > 0:
                                 if str(result["method.{}".format(k)][q]) == "sensedisambiguation.replace":
                                     matrix[3][0] = matrix[3][0] + 1
                                 if str(result["method.{}".format(k)][q]) == "wordnet.topn3.replace":
                                     matrix[3][1] = matrix[3][1] + 1
                                 if str(result["method.{}".format(k)][q]) == "tagmee.topn3":
                                     matrix[3][2] = matrix[3][2] + 1
                                 if str(result["method.{}".format(k)][q]) == "word2vec.topn3.replace":
                                     matrix[3][3] = matrix[3][3] + 1
                                 if str(result["method.{}".format(k)][q]) in ("glove.topn3.replace", "glove.topn3"):
                                     matrix[3][4] = matrix[3][4] + 1
                                 if str(result["method.{}".format(k)][q]) in ("wiki.topn3", "wiki.topn3.replace"):
                                     matrix[3][5] = matrix[3][5] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.krovetz":
                                     matrix[3][6] = matrix[3][6] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.porter":
                                     matrix[3][7] = matrix[3][7] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.porter2":
                                     matrix[3][8] = matrix[3][8] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.trunc4":
                                     matrix[3][9] = matrix[3][9] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.paicehusk":
                                     matrix[3][10] = matrix[3][10] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.trunc5":
                                     matrix[3][11] = matrix[3][11] + 1
                                 if str(result["method.{}".format(k)][q]) == "conceptluster.topn5.3.bm25":
                                     matrix[3][12] = matrix[3][12] + 1
                                 if str(result["method.{}".format(k)][q]) == "termluster.topn5.3.bm25":
                                     matrix[3][13] = matrix[3][13] + 1
                                 if str(result["method.{}".format(k)][q]) == "relevancefeedback.topn10.bm25":
                                     matrix[3][14] = matrix[3][14] + 1
                                 if str(result["method.{}".format(k)][q]) == "docluster.topn10.3.bm25":
                                     matrix[3][15] = matrix[3][15] + 1
                if str(classes["Class"][c]) == "Advice":
                         for k in range(j,0,-1):
                             if k > 0:
                                 if str(result["method.{}".format(k)][q]) == "sensedisambiguation.replace":
                                     matrix[4][0] = matrix[4][0] + 1
                                 if str(result["method.{}".format(k)][q]) == "wordnet.topn3.replace":
                                     matrix[4][1] = matrix[4][1] + 1
                                 if str(result["method.{}".format(k)][q]) == "tagmee.topn3":
                                     matrix[4][2] = matrix[4][2] + 1
                                 if str(result["method.{}".format(k)][q]) == "word2vec.topn3.replace":
                                     matrix[4][3] = matrix[4][3] + 1
                                 if str(result["method.{}".format(k)][q]) in ("glove.topn3.replace", "glove.topn3"):
                                     matrix[4][4] = matrix[4][4] + 1
                                 if str(result["method.{}".format(k)][q]) in ("wiki.topn3", "wiki.topn3.replace"):
                                     matrix[4][5] = matrix[4][5] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.krovetz":
                                     matrix[4][6] = matrix[4][6] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.porter":
                                     matrix[4][7] = matrix[4][7] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.porter2":
                                     matrix[4][8] = matrix[4][8] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.trunc4":
                                     matrix[4][9] = matrix[4][9] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.paicehusk":
                                     matrix[4][10] = matrix[4][10] + 1
                                 if str(result["method.{}".format(k)][q]) == "stem.trunc5":
                                     matrix[4][11] = matrix[4][11] + 1
                                 if str(result["method.{}".format(k)][q]) == "conceptluster.topn5.3.bm25":
                                     matrix[4][12] = matrix[4][12] + 1
                                 if str(result["method.{}".format(k)][q]) == "termluster.topn5.3.bm25":
                                     matrix[4][13] = matrix[4][13] + 1
                                 if str(result["method.{}".format(k)][q]) == "relevancefeedback.topn10.bm25":
                                     matrix[4][14] = matrix[4][14] + 1
                                 if str(result["method.{}".format(k)][q]) == "docluster.topn10.3.bm25":
                                     matrix[4][15] = matrix[4][15] + 1
    print(matrix)
    with open("../../qe/output/trec2009mq/stat/test_matrix_result.csv", "w") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(matrix)

def heatmap():
    CSVData = open("../../qe/output/trec2009mq/stat/test_matrix_result.csv")
    matrix_result = np.loadtxt(CSVData, delimiter=",")
    x_axis_labels = ["SenseDisambiguation", "Wordnet", "Tagmee","Word2Vec","Glove","Wiki","KrovetzStemmer","PorterStemmer","Porter2Stemmer","Trunc4Stemmer","PaiceHuskStemmer","Trunc5Stemmer","Conceptluster","Termluster","RelevanceFeedback","Docluster"]  # labels for x-axis
    y_axis_labels = ["Info_Close", "Info_Open", "Navigational", "Resource", "Advice"]  # labels for y-axis
    # create seabvorn heatmap with required labels
    sns.heatmap(matrix_result, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    # plt.pcolormesh( Array2d_result, cmap = 'coolwarm')
    plt.title("Improvement of the initial queries \n by different query expanders given a query type.", fontsize=12)
    plt.savefig("../qe/prep/trec2009mq/stat/heatmap.png", bbox_inches='tight', dpi=100)
    plt.show()

if __name__ == "__main__":
    bulding_matrix()
    heatmap()


