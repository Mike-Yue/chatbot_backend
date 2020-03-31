"""
Script for splitting data into training and testing (50/50)

Input should be the Testing.csv

Since Testing.csv is sorted by disease, consider the set of examples that make up a single
diesease one by one. For each example, measure its average? cosine similarity to others in its group
Ideally, the examples that are "middle" in similarity should be the ones in the testing set

"""

import csv
import sys
import math
from sklearn.metrics.pairwise import cosine_similarity

def main():
    test_split = 0.3

    csv_in_file = sys.argv[1]
    training_out_file = "CuratedTraining.csv"
    testing_out_file = "CuratedTesting.csv"

    train_out_file = open(training_out_file, 'w', newline='')
    train_writer = csv.writer(train_out_file)

    test_out_file = open(testing_out_file, 'w', newline='')
    test_writer = csv.writer(test_out_file)

    with open(csv_in_file, 'r') as in_file:
        reader = csv.reader(in_file)
        header = next(reader)
        train_writer.writerow(header)
        test_writer.writerow(header)

        curr_disease = ''
        curr_rows = [] # Rows associated with curr_disease
        for row in reader:
            vec = row[0:len(row)-1]
            disease = row[len(row)-1]
            if disease != curr_disease: #Reset, split current set
                split_helper(curr_disease, curr_rows, train_writer, test_writer, test_split)
                curr_disease = disease
                curr_rows = []
            curr_rows.append(vec)
        split_helper(curr_disease, curr_rows, train_writer, test_writer, test_split)


# Splits the rows of the given disease based on cosine similarity measures to one another
# Attempts to do a (100-test_split*100) / test_split*100 split to the training and testing set
# Rows with the middle most average similarity to those given will be placed in testing set
# test_split is a number between 0 and 1 to determine how many entries are given to the test dataset
def split_helper(disease, rows, train_writer, test_writer, test_split):
    if len(rows) == 0:
        return
    index_sim = []
    for i in range(len(rows)):
        base_row = rows[i]
        total_sim = 0
        for j in range(len(rows)):
            comp_row = rows[j]
            sim = cosine_similarity([base_row], [comp_row])[0][0]
            total_sim += sim
        avg_sim = total_sim / len(rows)
        index_sim.append((i, avg_sim))
    # Sort index_sim by avg_sim
    sorted_index_sim = sorted(index_sim, key=lambda x: x[1], reverse = True)
    test_n = math.floor(test_split * len(rows)) # Number of rows to put in out_test (CHANGE T)
    train_n = len(rows) - test_n # Number of rows to put in out_train
    base = math.floor(train_n / 2) # Take the middle test_n rows, take range from train_n/2 : train_n/2 + test_n
    for i in range(len(sorted_index_sim)):
        index = sorted_index_sim[i][0]
        if (i >= base and i < base + test_n): # Test row
            test_writer.writerow(rows[index] + [disease])
        else: # Train row
            train_writer.writerow(rows[index] + [disease])


if __name__ == "__main__":
    main()