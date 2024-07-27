# abonus.py

# template for Bonus Assignment, Artificial Intelligence Survey, CMPT 310 D200
# Spring 2021, Simon Fraser University

# author: Jens Classen (jclassen@sfu.ca)

import csv
from learning import *

def generate_restaurant_dataset(size=100):
    """
    Generate a data set for the restaurant scenario, using a numerical
    representation that can be used for neural networks. Examples will
    be newly created at random from the "real" restaurant decision
    tree.
    :param size: number of examples to be included
    """
    numeric_examples = []
    syn_res = SyntheticRestaurant(size)

    for i in range(size):
        temp_example = []
        for j in range(11):
            if syn_res.examples[i][j] == 'Yes':
                temp_example.append(1)
            elif syn_res.examples[i][j] == 'No':
                temp_example.append(0)
            elif syn_res.examples[i][j] == 'None':
                temp_example.append(0)
            elif syn_res.examples[i][j] == 'Some':
                temp_example.append(1)
            elif syn_res.examples[i][j] == 'Full':
                temp_example.append(2)
            elif syn_res.examples[i][j] == '$':
                temp_example.append(0)
            elif syn_res.examples[i][j] == '$$':
                temp_example.append(1)
            elif syn_res.examples[i][j] == '$$$':
                temp_example.append(2)
            elif syn_res.examples[i][j] == '0-10':
                temp_example.append(0)
            elif syn_res.examples[i][j] == '10-30':
                temp_example.append(1)
            elif syn_res.examples[i][j] == '30-60':
                temp_example.append(2)
            elif syn_res.examples[i][j] == '>60':
                temp_example.append(3)
            elif syn_res.examples[i][j] == 'Burger':
                temp_example.append(1)
                temp_example.append(0)
                temp_example.append(0)
                temp_example.append(0)
            elif syn_res.examples[i][j] == 'French':
                temp_example.append(0)
                temp_example.append(1)
                temp_example.append(0)
                temp_example.append(0)
            elif syn_res.examples[i][j] == 'Italian':
                temp_example.append(0)
                temp_example.append(0)
                temp_example.append(1)
                temp_example.append(0)
            elif syn_res.examples[i][j] == 'Thai':
                temp_example.append(0)
                temp_example.append(0)
                temp_example.append(0)
                temp_example.append(1)
        numeric_examples.append(temp_example)
    
    return DataSet(name='restaurant_numeric',
                   target='Wait',
                   examples=numeric_examples,
                   attr_names='Alternate Bar Fri/Sat Hungry Patrons Price Raining Reservation Burger French Italian Thai WaitEstimate Wait')

def nn_cross_validation(dataset, hidden_units, epochs=100, k=10):
    """
    Perform k-fold cross-validation. In each round, train a
    feed-forward neural network with one hidden layer. Returns the
    error ratio averaged over all rounds.
    :param dataset:      the data set to be used
    :param hidden_units: the number of hidden units (one layer) of the neural nets to be created
    :param epochs:       the maximal number of epochs to be performed in a single round of training
    :param k:            k-parameter for cross-validation 
                         (do k many rounds, use a different 1/k of data for testing in each round) 
    """

    hidden_layer = [hidden_units]
    fold_error_V = 0
    fold_error_T = 0
    random.shuffle(dataset.examples)
    size = len(dataset.examples)
    data_example = dataset.examples

    for fold in range(k):
        train_data, val_data = train_test_split(dataset, fold * (size // k), (fold + 1) * (size // k))
        dataset.examples = train_data
        heuristic = NeuralNetLearner(dataset, hidden_layer)
        fold_error_T += err_ratio(heuristic, dataset, train_data)
        fold_error_V += err_ratio(heuristic, dataset, val_data)
        # reverting back to original once test is completed
        dataset.examples = data_example

    return (fold_error_T / k + fold_error_V / k) / 2


N = 100   # number of examples to be used in experiments
k = 5   # k parameter
epochs = 100   # maximal number of epochs to be used in each training round
size_limit = 15   # maximal number of hidden units to be considered

# generate a new, random data set
# use the same data set for all following experiments
dataset = generate_restaurant_dataset(N)

# try out possible numbers of hidden units
for hidden_units in range(1,size_limit+1):
    # do cross-validation
    error = nn_cross_validation(dataset=dataset,
                                hidden_units=hidden_units,
                                epochs=epochs,
                                k=k)
    # report size and error ratio
    print("Size " + str(hidden_units) + ":", error)
