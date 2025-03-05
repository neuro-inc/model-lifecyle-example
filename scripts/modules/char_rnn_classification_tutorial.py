from io import open
import glob
import json
import os
from pathlib import Path
import random
import string 
import time
from typing import Dict, List, Optional, Tuple
import unicodedata
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mlflow

formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

ALLOWED_CHARACTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALLOWED_CHARACTERS) 

# -*- coding: utf-8 -*-
"""
NLP From Scratch: Classifying Names with a Character-Level RNN
**************************************************************
**Author**: `Sean Robertson <https://github.com/spro>`_

This tutorials is part of a three-part series:

* `NLP From Scratch: Classifying Names with a Character-Level RNN <https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html>`__
* `NLP From Scratch: Generating Names with a Character-Level RNN <https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html>`__
* `NLP From Scratch: Translation with a Sequence to Sequence Network and Attention <https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`__

"""

######################################################################
# Preparing Torch 
# ==========================
#
# Set up torch to default to the right device use GPU acceleration depending on your hardware (CPU or CUDA). 
#

import torch 

# Check if CUDA is available
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

torch.set_default_device(DEVICE)
logger.info(f"Using device = {torch.get_default_device()}")

######################################################################
# Preparing the Data
# ==================
#
# Download the data from `here <https://download.pytorch.org/tutorial/data.zip>`__ 
# and extract it to the current directory.
#
# Included in the ``data/names`` directory are 18 text files named as
# ``[Language].txt``. Each file contains a bunch of names, one name per
# line, mostly romanized (but we still need to convert from Unicode to
# ASCII).
#
# The first step is to define and clean our data. Initially, we need to convert Unicode to plain ASCII to 
# limit the RNN input layers. This is accomplished by converting Unicode strings to ASCII and allowing only a small set of allowed characters. 


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427    
def unicodeToAscii(s):
    global ALLOWED_CHARACTERS
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALLOWED_CHARACTERS
    )



######################################################################
# Turning Names into Tensors
# ==========================
# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    global ALLOWED_CHARACTERS
    return ALLOWED_CHARACTERS.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    global N_LETTERS
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


class NamesDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir #for provenance of the dataset
        self.load_time = time.localtime #for provenance of the dataset 
        labels_set = set() #set of all classes

        self.data = []
        self.data_tensors = []
        self.labels = [] 
        self.labels_tensors = [] 

        #read all the ``.txt`` files in the specified directory
        text_files = glob.glob(os.path.join(data_dir, '*.txt'))                           
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            for name in lines: 
                self.data.append(name)
                self.data_tensors.append(lineToTensor(name))
                self.labels.append(label)

        #Cache the tensor representation of the labels 
        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx] 

        return label_tensor, data_tensor, data_label, data_item


#########################
#Here we can load our example data into the ``NamesDataset``
def load_data(data_dir: str = "data/names") -> NamesDataset:
    logger.info("Starting to load data")
    alldata = NamesDataset(data_dir)
    logger.info(f"loaded {len(alldata)} items of data")
    logger.info(f"example = {alldata[0]}")
    logger.info("Finished loading data")
    return alldata

#########################
#Using the dataset object allows us to easily split the data into train and test sets. Here we create a 80/20 
# split but the ``torch.utils.data`` has more useful utilities. Here we specify a generator since we need to use the 
#same device as PyTorch defaults to above. 

def train_test_split(data: NamesDataset) -> Tuple[NamesDataset, NamesDataset]:
    logger.info("Starting to split data")
    train_set, test_set = torch.utils.data.random_split(data, [.85, .15], generator=torch.Generator(device=DEVICE).manual_seed(2024))
    logger.info(f"train examples = {len(train_set)}, validation examples = {len(test_set)}")
    logger.info("Finished spliting data")
    return train_set, test_set # type: ignore

#########################
# Now we have a basic dataset containing **20074** examples where each example is a pairing of label and name. We have also 
#split the dataset into training and testing so we can validate the model that we build. 


######################################################################
# Creating the Network
# ====================


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)

        return output


###########################
# We can then create an RNN with 57 input nodes, 128 hidden nodes, and 18 outputs:
def get_rnn(data: NamesDataset) -> Tuple[CharRNN, Dict]:
    global N_LETTERS
    logger.info("Gettint RNN")
    params = {
        "n_hidden": 128,
        "n_categories": len(data.labels_uniq),
        "categories": data.labels_uniq
    }
    rnn = CharRNN(N_LETTERS, hidden_size=params["n_hidden"], output_size=params["n_categories"])
    return rnn, params


######################################################################
# After that we can pass our Tensor to the RNN to obtain a predicted output. Subsequently,  
# we use a helper function, ``label_from_output``, to derive a text label for the class.

def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i


######################################################################
#
# Training
# ========


######################################################################
# Training the Network
# --------------------
#
# Now all it takes to train this network is show it a bunch of examples,
# have it make guesses, and tell it if it's wrong.
#
# We do this by defining a ``train()`` function which trains the model on a given dataset using minibatches. RNNs 
# RNNs are trained similarly to other networks; therefore, for completeness, we include a batched training method here.
# The loop (``for i in batch``) computes the losses for each of the items in the batch before adjusting the 
# weights. This operation is repeated until the number of epochs is reached. 

def train(rnn, training_data, n_epoch = 10, n_batch_size = 64, report_every = 50, learning_rate = 0.2, criterion = nn.NLLLoss()):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    logger.info("Starting training")
    
    # Log parameters
    mlflow.log_params({
        "n_epoch": n_epoch,
        "batch_size": n_batch_size,
        "learning_rate": learning_rate,
        "dataset_size": len(training_data)
    })
    
    # Existing training setup code
    current_loss = 0
    all_losses = []
    rnn.train() 
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    start = time.time()
    logger.info(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1): 
        rnn.zero_grad() # clear the gradients 

        # create some minibatches
        # we cannot use dataloaders because each of our names is a different length
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) //n_batch_size )

        for idx, batch in enumerate(batches): 
            batch_loss = 0
            for i in batch: #for each example in this batch
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)
        
        all_losses.append(current_loss / len(batches) )
        # Log metrics
        mlflow.log_metric("avg_loss", all_losses[-1], step=iter)
        
        if iter % report_every == 0:
            logger.info(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0
    
    # Log training time
    training_time = time.time() - start
    if mlflow.active_run():
        mlflow.log_metric("training_time", training_time)
    
    # Log the model
    if mlflow.active_run():
        mlflow.pytorch.log_model(rnn, "model")
        
    logger.info("Finished training")
    return all_losses

##########################################################################
# We can now train a dataset with minibatches for a specified number of epochs. The number of epochs for this 
# example is reduced to speed up the build. You can get better results with different parameters.



######################################################################
# Evaluating the Results
# ======================
#
# To see how well the network performs on different categories, we will
# create a confusion matrix, indicating for every actual language (rows)
# which language the network guesses (columns). To calculate the confusion
# matrix a bunch of samples are run through the network with
# ``evaluate()``, which is the same as ``train()`` minus the backprop.
#

def evaluate(rnn, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))
    
    rnn.eval() #set to eval mode
    with torch.no_grad(): # do not record the gradients during eval phase
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = rnn(text_tensor)   
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(len(classes)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy()) #numpy uses cpu here so we need to use a cpu version
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()
    fig.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

def load_model(weights_file: Path, params: Dict) -> CharRNN:
    rnn = CharRNN(
        input_size=N_LETTERS,
        hidden_size=params["n_hidden"],
        output_size=params["n_categories"],
    )
    rnn.load_state_dict(torch.load(weights_file, weights_only=True, map_location=DEVICE))
    return rnn

def predict(
    name,
    n_predictions: int,
    params: Dict,
    rnn: Optional[CharRNN] = None,
    weights_file: Optional[Path] = None,
) -> List[Tuple[torch.Tensor, str]]:
    if rnn is None:
        if weights_file is None:
            raise ValueError("Either rnn or weights_file must be provided")
        rnn = load_model(weights_file, params)

    logger.info(f"{N_LETTERS=}, {params['n_hidden']=}, {params['n_categories']=}")
    with torch.no_grad():
        line = Variable(lineToTensor(name).to(DEVICE))
        output = rnn(line)

    logger.info(output)
    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        predictions.append([value, params["categories"][category_index]])

    return predictions
######################################################################
# Save the model
# =============
#
# Save the trained model to disk for later use

def save_model(rnn: CharRNN, params: Dict, models_dir: str = "models") -> None:
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Save the model locally
    model_path = os.path.join(models_dir, 'weights.pt')
    torch.save(rnn.state_dict(), model_path)
    logger.info(f'Model saved to {model_path}')
    with open(os.path.join(models_dir, 'params.json'), 'w') as f:
        json.dump(params, f)
    
    # Log model artifact path to MLflow
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(os.path.join(models_dir, 'params.json'))


def train_eval_save(data_dir: str = "data/names", models_dir: str = "models"):
    alldata = load_data(data_dir)
    with mlflow.start_run():
        train_set, test_set = train_test_split(alldata)
        rnn, params = get_rnn(alldata)
        logger.info(f"Training params: {params}")
        start = time.time()
        all_losses = train(rnn, train_set, n_epoch=27, learning_rate=0.15, report_every=5)
        end = time.time()
        logger.info(f"Training took {end-start}s")

        ######################################################################
        # Plotting the Results
        # --------------------
        #
        # Plotting the historical loss from ``all_losses`` shows the network
        # learning:
        #

        plt.figure()
        plt.plot(all_losses)
        plt.show()
        plt.savefig("training_loss.png")
        mlflow.log_artifact("training_loss.png")

        # Evaluate
        evaluate(rnn, test_set, classes=alldata.labels_uniq)

        save_model(rnn, params, models_dir)

    return rnn

