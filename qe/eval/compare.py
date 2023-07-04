import re
import optparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_tValue_cohensD(map1, map2):
    # Calculate the mean and standard deviation
    mean_diff = map2 - map1
    std_diff = np.sqrt((map1 ** 2 + map2 ** 2) / 2)

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
        map1 = row[col1]
        new_line = [row.qid, map1]

        for index, map2 in enumerate(col2_list):
            # Check empty cells
            t_value, cohens_d = calculate_tValue_cohensD(map1=map1, map2=row[map2])

            # New row of the output file csv file
            new_line.append(row[map2])  # col 2 mAP
            new_line.append(t_value)  # t_value
            new_line.append(cohens_d)  # cohens_d

        csv_result.loc[len(csv_result)] = new_line

    # write dataFrame to a CSV file
    csv_result.to_csv(outfile_path + '.csv')


def plot_result(infile_df, col1, col2):
    # Map of the main column
    map1 = infile_df[[word for word in list(infile_df.columns) if word.startswith(col1) and word.endswith('map')][0]].tolist()
    map1 = [0 if str(item) == 'nan' else item for item in map1]

    # Plot the difference between maps of col1 and col2
    for word in list(infile_df.columns):
        if word.startswith(col2) and word.endswith("map"):
            map2 = [0 if str(item) == 'nan' else item for item in infile_df[word]]
            diff = np.subtract(map2, map1)  # Delta map
            plt.plot(diff, label=word.replace(word[word.find('.'):], ''))

    plt.legend()
    plt.show()


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


if __name__ == '__main__':
    columns = get_input()

    # Constant
    infile_path = '../qe/output/robust04/topics.robust04.bm25.map.all.csv'
    outfile_path = '../qe/output/compare/'

    infile_df = pd.read_csv(infile_path)
    infile_df = infile_df.reset_index()  # making sure indexes pair with number of rows

    outfile_path += f'compare.mAP.{columns.col1}.{columns.col2}'

    # compare_mAP_each_row(infile_df=infile_df, outfile_path=outfile_path, col1=columns.col1, col2=columns.col2)
    # compare_mAP_all_row(infile_df=infile_df, outfile_path=outfile_path, col1=columns.col1, col2=columns.col2)
    plot_result(infile_df=infile_df, col1=columns.col1, col2=columns.col2)
