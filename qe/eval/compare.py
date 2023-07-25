import re
import optparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
sys.path.extend(['../qe/output'])

def calculate_tValue_cohensD(map1, map2, i):
    # Calculate the mean and standard deviation
    mean_diff = map2 - map1
    std_diff = np.sqrt((map1 ** 2 + map2 ** 2) / 2)

    if std_diff == 0:
        cohens_d = 0
        t_value = 0
    else:
        # Calculate Cohen's d
        cohens_d = mean_diff / std_diff

        n = 1  # Number of observations (in this case, each value is a single observation)
        std_error = std_diff / np.sqrt(n)
        t_value = mean_diff / std_error

    return t_value, cohens_d


def compare_mAP_each_row(infile_df, outfile_path, col1, col2):
    # Initialize header
    headers = ['qid']
    col2_list = []
    for word in list(infile_df.columns):
        if word.startswith(col2) and word.endswith("map"):
            headers.append(word)
            headers.append('t-value')
            headers.append('cohens-d')
            col2_list.append(word)
        elif word.startswith(col1) and word.endswith("map"):
            # Make sure the main col that wants to do the comparison with, is placed at the second col.
            # The first column that match the col1 input is considered for main col.
            headers.insert(1, word)
            col1 = word

    # Output csv file
    csv_result = pd.DataFrame(columns=headers)

    for index, row in infile_df.iterrows():
        # Initialize new line
        # If a list of names, starting with col1, is returned, only the first one is considered!
        map1 = 0 if str(row[col1]) == 'nan' else float(row[col1])
        new_line = [row.qid, map1]

        for index, map2 in enumerate(col2_list):
            m2 = 0 if str(row[map2]) == 'nan' else float(row[map2])
            t_value, cohens_d = calculate_tValue_cohensD(map1=map1, map2=m2, i=index)

            # New row of the output file csv file
            new_line.append(row[map2])  # col 2 mAP
            new_line.append(t_value)  # t_value
            new_line.append(cohens_d)  # cohens_d

        csv_result.loc[len(csv_result)] = new_line

    # write dataFrame to a CSV file
    csv_result.to_csv(outfile_path + '.csv')


def plot_result(infile_df, outfile, col1, col2):
    # Map of the main column
    map1 = infile_df[[word for word in list(infile_df.columns) if word.startswith(col1) and word.endswith('map')][0]].tolist()
    map1 = [0 if str(item) == 'nan' else float(item) for item in map1]

    # Plot the difference between maps of col1 and col2
    for word in list(infile_df.columns):
        if word.startswith(col2) and word.endswith("map"):
            map2 = [0 if str(item) == 'nan' else float(item) for item in infile_df[word]]
            diff = np.subtract(map2, map1)  # Delta map
            plt.plot(diff, label=word.replace(word[word.find('.'):], ''))

    plt.legend()

    # Saving the figure.
    plt.savefig(outfile + ".jpg")


def compare_mAP_all_row(infile_df, outfile_path, col1, col2):
    with open(outfile_path + '.txt', 'w') as file:
        for word in list(infile_df.columns):
            if (word.startswith(col2) or word.startswith(col1)) and word.endswith("map"):
                map = infile_df[word].tolist()
                mean = np.mean([0 if str(item) == 'nan' else item for item in map])
                file.write(f'{word} mean of mAP is: {mean}\n')


def get_input():
    # Instantiate the parser
    parser = optparse.OptionParser()
    parser.add_option('--column1',
                      action="store", dest="col1",
                      help="Query string, the main column you want to do the comparison",
                      default="abstractqueryexpansion")
    parser.add_option('--column2',
                      action="store", dest="col2",
                      help="Query string, Name of the expander or column(s) (this name can refer to multiple columns) "
                           "with want to compare to the first column",
                      default="backtranslation")


    options, args = parser.parse_args()

    return options

def analysis(infile_df, outfile_path):
    with open(outfile_path + ".txt", "w") as file:
        for column in range(3, len(infile_df.columns), 4):

            # output language name
            file.write("<lang> " + str(infile_df.columns.tolist()[column + 1])[16:])
            file.write("\n")

            for row in range(len(infile_df.iloc[:, column])):

                # output format: <qid> <abstractqueryexpansion> <abstractqueryexpansion.bm25.map> <backtranslation_pes_arab> <semsim> <backtranslation_pes_arab.bm25.map> <subtraction>
                file.write(str(infile_df.iloc[row, column]) +
                           "\t" + str(infile_df.iloc[row, 1]) +
                           "\t" + str(infile_df.iloc[row, 2]) +
                           "\t" + str(infile_df.iloc[row, column + 1]) +
                           "\t" + str(infile_df.iloc[row, column + 2]) +
                           "\t" + str(infile_df.iloc[row, column + 3]) +
                           "\t" + str(infile_df.iloc[row, column + 3] - infile_df.iloc[row, 2]))
                file.write("\n")

def find_best_map(input_file):
    #
    with open(input_file, 'w') as file:
        pass

if __name__ == '__main__':
    columns = get_input()

    # Constant
    infile_path = '../output/gov2/topics.gov2.701-850.bm25.map.all.csv'
    corpus = 'gov2'
    outfile_path = '../output/compare/' + corpus + '/'

    if not os.path.exists(outfile_path):
        os.makedirs(outfile_path)

    infile_df = pd.read_csv(infile_path)
    infile_df = infile_df.reset_index()  # making sure indexes pair with number of rows

    outfile_path += f'compare.{corpus}.mAP.{columns.col1}.{columns.col2}'

    compare_mAP_each_row(infile_df=infile_df, outfile_path=outfile_path, col1=columns.col1, col2=columns.col2)
    compare_mAP_all_row(infile_df=infile_df, outfile_path=outfile_path, col1=columns.col1, col2=columns.col2)
    plot_result(infile_df=infile_df, outfile=outfile_path, col1=columns.col1, col2=columns.col2)
