import argparse
import torch
import torch.nn as nn
import numpy as np
from LSTNet import LSTNetForClassification
from utils import *
import Optim
from unimib_shar_adl_load_dataset import unimib_load_dataset


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def get_shapes(np_arr_list):
    """Returns text, each line is shape and dtype for numpy array in list
       example: print(get_shapes([X_train, X_test, y_train, y_test]))"""
    shapes = ""
    for i in np_arr_list:
        my_name = namestr(i,globals())
        shapes += (my_name[0] + " shape is " + str(i.shape) \
            + " data type is " + str(i.dtype) + "\n")
    return shapes


# Define a new evaluation function for classification
def evaluate_classification(data, X, Y, model, criterion, batch_size):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X)
        loss = criterion(output, Y)

        total_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(Y).sum().item()
        total_samples += X.size(0)

    accuracy = correct / total_samples
    return total_loss, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Time series classification')
    parser.add_argument('--model', type=str, default='LSTNet', help='')
    parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units')
    parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units')
    parser.add_argument('--num_classes', type=int, default=9, help='number of classes')  # Adjust the number of classes
    parser.add_argument('--window', type=int, default=24 * 7, help='window size')  # Add 'window' parameter
    parser.add_argument('--classification_loss', type=str, default='cross_entropy', help='classification loss function (e.g., cross_entropy)')
    parser.add_argument('--cuda', action='store_true', default=False,
                    help='Enable CUDA for GPU acceleration')

    # Add other arguments as needed
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()  # Check if CUDA is available and the flag is set


    if args.classification_loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Unsupported classification loss function")

    dataset = "UniMiB SHAR"
    datasetfn = 'mobiact_adl_load_dataset.py'
    full_filename = 'unimib_shar_adl_load_dataset.py'
   
    #x_train, y_train, x_test, y_test = unimib_load_dataset(incl_val_group = False)
    x_train, y_train, x_validate, y_validate, x_test, y_test = unimib_load_dataset(incl_val_group = True)
    t_names = ['StandingUpFS','StandingUpFL','Walking','Running','GoingUpS','Jumping','GoingDownS','LyingDownFS','SittingDown']

    train_ratio = 0.6
    valid_ratio = 0.2
    cuda = True
    horizon = 12 
    window = 24
    normalize = 2

    data = Data_utility(x_train, train_ratio, valid_ratio, cuda, horizon, window, normalize)

    print(get_shapes([x_train, y_train, x_validate, y_validate, x_test, y_test]))

    # You may need to preprocess the dataset (e.g., normalization, reshaping) to fit the model

    # Initialize and load your LSTNetForClassification model here
    model = LSTNetForClassification(args, data )  # Modify this according to your model

    # Train your model and perform validation
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            # Training loop
            train_loss = train(x_train, x_train.train[0], x_train.train[1], model, criterion, optim, args.batch_size)

            # Validation loop
            val_loss, val_accuracy = evaluate_classification(x_train, x_train.valid[0], x_train.valid[1], model, criterion, args.batch_size)

            print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid_loss {:5.4f} | valid_accuracy {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_accuracy))

        # Load the best saved model and evaluate on the test set
        with open(args.save, 'rb') as f:
            model = torch.load(f)
        test_loss, test_accuracy = evaluate_classification(x_train, x_test.test[0], x_test.test[1], model, criterion, args.batch_size)
        print("test_loss {:5.4f} | test_accuracy {:5.4f}".format(test_loss, test_accuracy))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
