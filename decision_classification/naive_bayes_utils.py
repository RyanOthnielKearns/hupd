# PyTorch
import torch

# standard
import random
import numpy as np
import collections
from tqdm import tqdm

# sklearn 
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

# Fixing the random seeds
RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Number of classes (ACCEPTED and REJECTED)
CLASSES = 2
CLASS_NAMES = [i for i in range(CLASSES)]

# Calculate TOP1 accuracy
def measure_accuracy(outputs, labels):
    preds = np.argmax(outputs, axis=1).flatten()
    labels = labels.flatten()
    correct = np.sum(preds == labels)
    c_matrix = confusion_matrix(labels, preds, labels=CLASS_NAMES)
    return correct, len(labels), c_matrix

# Create a BoW (Bag-of-Words) representation
def text2bow(input, vocab_size):
    arr = []
    for i in range(input.shape[0]):
        query = input[i]
        features = [0] * vocab_size
        for j in range(query.shape[0]):
            features[query[j]] += 1 # todo: for Multinomial (initially +1)
        arr.append(features)
    return np.array(arr)

# Evaluation procedure (for the Naive Bayes models)
def validation_naive_bayes(data_loader, model, vocab_size, name='validation', write_file=None, pad_id=-1):
    total_loss = 0.
    total_correct = 0
    total_sample = 0
    total_confusion = np.zeros((CLASSES, CLASSES))
    
    # Loop over all the examples in the evaluation set
    for i, batch in enumerate(tqdm(data_loader)):
        input, label = batch['input_ids'], batch['output']
        input = text2bow(input, vocab_size)
        input[:, pad_id] = 0
        logit = model.predict_log_proba(input)
        label = np.array(label.flatten()) 
        correct_n, sample_n, c_matrix = measure_accuracy(logit, label)
        total_confusion += c_matrix
        total_correct += correct_n
        total_sample += sample_n
    print(f'*** Accuracy on the {name} set: {total_correct/total_sample}')
    print(f'*** Confusion matrix:\n{total_confusion}')
    if write_file:
        write_file.write(f'*** Accuracy on the {name} set: {total_correct/total_sample}\n')
        write_file.write(f'*** Confusion matrix:\n{total_confusion}\n')
    return total_loss, float(total_correct/total_sample) * 100.


# Training procedure (for the Naive Bayes models)
def train_naive_bayes(data_loaders, tokenizer, vocab_size, version='Bernoulli', alpha=1.0, write_file=None, np_filename=None):
    pad_id = tokenizer.encode('[PAD]') # NEW
    print(f'Training a {version} Naive Bayes classifier (with alpha = {alpha})...')
    write_file.write(f'Training a {version} Naive Bayes classifier (with alpha = {alpha})...\n')

    # Bernoulli or Multinomial?
    if version == 'Bernoulli':
        model = BernoulliNB(alpha=alpha) 
    elif version == 'Multinomial':
        model = MultinomialNB(alpha=alpha) 
    
    # Loop over all the examples in the training set
    for i, batch in enumerate(tqdm(data_loaders[0])):
        input, decision = batch['input_ids'], batch['output']
        input = text2bow(input, vocab_size) # change text2bow(input[0], vocab_size)
        input[:, pad_id] = 0 # get rid of the paddings
        label = np.array(decision.flatten())
        # Using "partial fit", instead of "fit", to avoid any potential memory problems
        # model.partial_fit(np.array([input]), np.array([label]), classes=CLASS_NAMES)
        model.partial_fit(input, label, classes=CLASS_NAMES)
    
    print('\n*** Accuracy on the training set ***')
    validation_naive_bayes(data_loaders[0], model, vocab_size, 'training', write_file, pad_id)
    print('\n*** Accuracy on the validation set ***')
    validation_naive_bayes(data_loaders[1], model, vocab_size, 'validation', write_file, pad_id)
    
    # Save the log probabilities if np_filename is specified
    if np_filename:
        np.save(f'{np_filename}.npy', np.array(model.feature_log_prob_))