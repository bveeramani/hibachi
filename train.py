"""Implements functions for training a model.

usage: train.py [-h] [--test TEST_DATASET] [--batches BATCH_SIZE]
                [--epochs NUM_EPOCHS] [--disable-cuda]
                train

Train model

positional arguments:
  train                 path to trainining dataset

optional arguments:
  -h, --help            show this help message and exit
  --test TEST_DATASET   path to testing dataset
  --batches BATCH_SIZE  number of samples to propogate (default: 64)
  --epochs NUM_EPOCHS   number of passes through dataset (default: 32)
  --disable-cuda        disable CUDA support
"""
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import OteyP450
from models import *

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCH_SIZE = 64
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ModelType = SupportVectorMachine
MODEL_ARGS = [8]
MODEL_KWARGS = {}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('train_dataset',
                        metavar='train',
                        help='path to trainining dataset')
    parser.add_argument('--test',
                        dest='test_dataset',
                        default=None,
                        help='path to testing dataset')
    parser.add_argument('--batches',
                        type=int,
                        dest='batch_size',
                        default=DEFAULT_BATCH_SIZE,
                        help='number of samples to propogate (default: 64)')
    parser.add_argument('--epochs',
                        type=int,
                        dest='num_epochs',
                        default=DEFAULT_EPOCH_SIZE,
                        help='number of passes through dataset (default: 32)')
    parser.add_argument('--disable-cuda',
                        dest='cuda_disabled',
                        action='store_true',
                        help='disable CUDA support')
    parser.add_argument('--save',
                        dest='save_model',
                        action='store_true',
                        help='save trained model')


    args = parser.parse_args()

    if not args.cuda_disabled and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = ModelType(*MODEL_ARGS, **MODEL_KWARGS)
    train_dataset = OteyP450(args.train_dataset)
    test_dataset = OteyP450(args.test_dataset) if args.test_dataset else None

    model = train(model,
                  train_dataset,
                  test_dataset=test_dataset,
                  batch_size=args.batch_size,
                  num_epochs=args.num_epochs,
                  device=device)

    if args.save_model:
        save_model(model, batch_size=args.batch_size, num_epochs=args.num_epochs)

    return 0


#pylint: disable=too-many-arguments, too-many-locals
def train(model,
          train_dataset,
          test_dataset=None,
          batch_size=DEFAULT_BATCH_SIZE,
          num_epochs=DEFAULT_EPOCH_SIZE,
          loss_func=nn.BCELoss(),
          device=DEFAULT_DEVICE):
    """Trains the specified model.

    If a test dataset is supplied, then this function will propogate over
    the test dataset every epoch to check the model accuracy.

    The specified model will always be trained using the SGD optimizer.
    """

    def debug_train(train_dataset, test_dataset, batch_size, num_epochs, device):
        """Prints information about the training hyperparameters."""
        assert train_dataset, "Expected non-empty dataset."

        start_time = datetime.datetime.now()
        print("Starting training at time %s." % start_time, end="\n\n")

        print("BATCH_SIZE=%d" % batch_size)
        print("NUM_EPOCHS=%d" % num_epochs)
        if torch.cuda.device_count() > 1:
            print("NUM_GPUS=%d" % torch.cuda.device_count())
        print("DEVICE=%s" % device, end="\n\n")

        features, label = train_dataset[0]
        print("INPUT_SHAPE=", features.shape, sep="")
        print("LABEL_SHAPE=", label.shape, sep="")
        if test_dataset:
            print("TEST_DATASET_SIZE=%d" % len(test_dataset))
        print("DATASET_SIZE=%d" % len(train_dataset), end="\n\n")

    dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    debug_train(train_dataset, test_dataset, batch_size, num_epochs, device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        model.train()
        for batch_number, (features, labels) in enumerate(dataloader):
            model.zero_grad()

            features = features.to(device)
            labels = labels.to(device)

            predictions = model(features)
            for i in range(len(labels)):
                if labels[i] == 0:
                    labels[i] = -1

            loss = loss_func(predictions, labels)
            # debug_batch(batch_number, epoch, loss)

            loss.backward()
            optimizer.step()

        if test_dataset:
            test(model, test_dataset, device=device, batch_size=batch_size * 2)

    return model


def test(model,
         dataset,
         batch_size=DEFAULT_DEVICE * 2,
         loss_func=nn.BCELoss(),
         device=DEFAULT_DEVICE):
    """Tests a model over the specified dataset."""

    def debug_test(loss, accuracy):
        """Prints results from testing a model over a dataset.

        Arguments:
            loss: A scalar tensor
            accuracy: A floating-point number between 0 and 1.
        """
        print("LOSS\tACCURACY")
        print("%.4f\t%.4f" % (loss.item(), accuracy), end="\n\n")

    dataloader = DataLoader(dataset, batch_size)
    batch_accuracies = []

    model.eval()
    with torch.no_grad():
        total_loss = 0
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            predictions = model(features)
            batch_accuracy = average_accuracy(predictions, labels)

            total_loss += loss_func(predictions, labels)
            batch_accuracies.append(batch_accuracy)

    average_batch_accuracy = sum(batch_accuracies) / len(batch_accuracies)
    debug_test(total_loss, average_batch_accuracy)


def average_accuracy(predictions, labels):
    """Calculates the average prediction accuracy.

    N is the number of predictions.

    Arguments:
        predictions: A tensor of shape N
        labels: A scalar

    Returns:
        The average prediction accuracy.
    """
    num_correct = 0

    for prediction, label in zip(predictions, labels):
        if prediction > 0.5 and label == 1 or prediction <= 0.5 and label == 0:
            num_correct += 1

    return num_correct / len(predictions)


def save_model(model, batch_size, num_epochs):
    """Saves a model's state dictionary to a .pt file

    An example of a file that might be produced by this function
    is 6132234.b64.e32.pt. The first part of the filename indicates the month,
    day, hour, and minute at which the model was saved. The second part
    indicates the batch size, and the third part indicates the number of epochs.

    Arguments:
        model: An nn.Module object
        batch_size: An integer representing how many batches the model was
            trained on
        num_epochs: An integer representing how many epochs the model was
            trained over
    """
    end_time = datetime.datetime.now()
    filename = "%s%s%s%s.b%de%d.pt" % (end_time.month, end_time.day,
                                       end_time.hour, end_time.minute,
                                       batch_size, num_epochs)
    print("Saving model to %s" % filename)
    torch.save(model.module.state_dict(), filename)


if __name__ == "__main__":
    main()
