import collections

import numpy as np

import util as util
import svm as svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    list_of_words = message.split(' ')
    normalized_list = [word.lower() for word in list_of_words]
    return normalized_list
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    dictionary = dict()
    for message in messages:
        words = get_words(message)
        words = set(words)
        for word in words:
            if word not in dictionary:
                dictionary[word]=1
            else:
                dictionary[word]+=1
    mapper = dict()
    key = 0
    for word in dictionary:
        if dictionary[word]>=5:
            mapper[word]=key
            key+=1
    
    return mapper
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    n = len(messages)
    m = max(list(word_dictionary.values()))
    matrix = np.zeros((n,m+1), dtype=int)
    for pos,message in enumerate(messages):
        words = get_words(message)
        for word in words:
            if word in word_dictionary:
                matrix[pos][word_dictionary[word]]+=1

    return matrix
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    phi1 = np.mean(labels)
    phi0 = 1-phi1
    n, m = matrix.shape
    phis = np.zeros(n)
    phix1 = np.sum(matrix[labels==1], axis=0)
    phix1 +=1 
    phix1 = 1/(m+n) * phix1
    phix0 = np.sum(matrix[labels==0], axis=0)
    phix0 +=1 
    phix0 = 1/(m+n) * phix0
    return ((phi0, phi1), (phix0, phix1))
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    phi, phix = model
    logit = np.zeros((2,len(matrix)))
    n,m = matrix.shape
    for k in range(2):
        for i in range(n):
            logit[k][i]=np.sum(matrix[i, :]@np.log(phix[k]) + (1-matrix[i, :]) @ np.log(1-phix[k]))
            logit[k][i]+=np.log(phi[k])
    prediction = np.greater(logit[1], logit[0])
    prediction.astype(int)
    return prediction
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    phi, phix = model
    indicativeness = np.log(phix[1]) - np.log(phix[0])
    indicators = np.argsort(indicativeness)[-5:][::-1]
    return [list(dictionary.keys())[list(dictionary.values()).index(code)] for code in indicators]
    # *** END CODE HERE ***

def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    maxacc=0
    maxradius = radius_to_consider[0]
    for radius in radius_to_consider:
        accuracy = np.mean(svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)==val_labels)
        if (accuracy>=maxacc):
            maxradius=radius
            maxacc=accuracy
    return maxradius
            
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('data/ds6_test.tsv')
    
    dictionary = create_dictionary(train_messages)

    util.write_json('output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    #np.savetxt('output/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    #np.savetxt('output/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {}% on the testing set'.format(naive_bayes_accuracy*100))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)
    
    #util.write_json('output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    #util.write_json('output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {}% on the testing set'.format(svm_accuracy*100))
    
if __name__ == "__main__":
    main()
