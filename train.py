"""Implements functions for training a model.

usage: train.py [-h] [--test TEST_DATASET] [--batches BATCH_SIZE]
                [--epochs NUM_EPOCHS] [--save FILENAME] [--disable-cuda]
                train

Train model

positional arguments:
  train                 path to trainining dataset

optional arguments:
  -h, --help            show this help message and exit
  --test TEST_DATASET   path to testing dataset
  --batches BATCH_SIZE  number of samples to propogate (default: 32)
  --epochs NUM_EPOCHS   number of passes through dataset (default: 64)
  --save FILENAME       save trained model (default "model.py")
  --disable-cuda        disable CUDA support
"""
import argparse
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import OteyP450
from models import LogisticRegressionModel

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCH_SIZE = 64
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ModelType = LogisticRegressionModel
MODEL_ARGS = [OteyP450.NUM_FEATURES]
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
    parser.add_argument('--save',
                        dest='save_filename',
                        metavar='FILENAME',
                        default=None,
                        help='save trained model (default "model.py")')
    parser.add_argument('--disable-cuda',
                        dest='cuda_disabled',
                        action='store_true',
                        help='disable CUDA support')

    args = parser.parse_args()

    if not args.cuda_disabled and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    train_dataset = OteyP450(args.train_dataset)
    test_dataset = OteyP450(args.test_dataset) if args.test_dataset else None

    model = ModelType(*MODEL_ARGS, **MODEL_KWARGS)

    model = train(model,
                  train_dataset,
                  batch_size=args.batch_size,
                  num_epochs=args.num_epochs,
                  device=device)

    if args.test_dataset:
        test(model, test_dataset, batch_size=args.batch_size * 2, device=device)

    if args.save_filename:
        save(model, args.save_filename)

    return 0


# pylint: disable=too-many-arguments, too-many-locals
def train(model,
          train_dataset,
          batch_size=DEFAULT_BATCH_SIZE,
          num_epochs=DEFAULT_EPOCH_SIZE,
          loss_func=nn.BCELoss(),
          device=DEFAULT_DEVICE,
          quiet=False):
    """Trains the specified model.

    The model is trained using the SGD optimizer.

    Arguments:
        model (nn.Module): A PyTorch model
        dataset (torch.utils.data.Dataset): The dataset to train on
        batch_size (int, optional): The training batch size.
        num_epochs (int, optional): The number of epochs to propogate over
        loss_func (callable, optional): The cost function.
        device (str): The device on which to run the test.
        quiet (bool, optional): If true, then this function will not print
            status messages to STDOUT.
    """

    def print_train_settings(train_dataset, batch_size, num_epochs, device):
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
        print("DATASET_SIZE=%d" % len(train_dataset), end="\n\n")

    def print_batch_results(epoch, batch, loss):
        if epoch == 0 and batch == 0:
            print("EPOCH\tBATCH\tLOSS")
        print("%d\t%d\t%.4f" % (epoch, batch, loss.item()))

    dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    if not quiet:
        print_train_settings(train_dataset, batch_size, num_epochs, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    if torch.cuda.device_count() > 1 and not device == "cpu":
        model = nn.DataParallel(model)
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch, (examples, labels) in enumerate(dataloader):
            model.zero_grad()

            examples = examples.to(examples)
            labels = labels.to(device)
            predictions = model(examples)
            # torch.Size([N, 1]) => torch.Size([N])
            predictions = torch.squeeze(predictions)
            loss = loss_func(predictions, labels)

            if not quiet:
                print_batch_results(epoch, batch, loss)

            loss.backward()
            optimizer.step()

    return model


def test(model,
         dataset,
         batch_size=DEFAULT_BATCH_SIZE * 2,
         device=DEFAULT_DEVICE,
         quiet=False):
    """Tests a model on the specified dataset.

    Arguments:
        model (nn.Module): A PyTorch model
        dataset (torch.utils.data.Dataset): The dataset to test on
        batch_size (int, optional): The desired test batch size.
        device (str): The device on which to run the test.
        quiet (bool, optional): If true, then this function will not print
            status messages to STDOUT.
    """

    def average_accuracy(predictions, labels):
        """Calculates the average classification accuracy.

        N is the number of predictions.

        Arguments:
            predictions: A 1-dimensional length-N tensor.
            labels: A 1-dimensional length-N tensor.

        Returns:
            The average prediction accuracy.
        """
        num_correct = 0

        for prediction, label in zip(predictions, labels):
            if prediction > 0.5 and label == 1 or prediction <= 0.5 and label == 0:
                num_correct += 1

        return num_correct / len(predictions)

    def print_test_results(accuracy):
        """Prints results from testing a model over a dataset.

        Arguments:
            accuracy (float): A number between 0 and 1.
        """
        print("ACCURACY\n%.4f" % accuracy, end="\n\n")

    dataloader = DataLoader(dataset, batch_size)
    batch_accuracies = []

    model.eval()
    with torch.no_grad():
        for obvservations, labels in dataloader:
            obvservations = obvservations.to(device)
            labels = labels.to(device)

            predictions = model(obvservations)
            batch_accuracy = average_accuracy(predictions, labels)

            batch_accuracies.append(batch_accuracy)

    average_batch_accuracy = sum(batch_accuracies) / len(batch_accuracies)

    if not quiet:
        print_test_results(average_batch_accuracy)

    return average_batch_accuracy


def save(model, filename):
    """Saves a model's state dictionary to a file.

    Arguments:
        model (nn.Module): The PyTorch model to save.
        filename (string): Desired model filename.
    """
    print("Saving model to %s" % filename)
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), filename)
    else:
        torch.save(model.state_dict(), filename)


if __name__ == "__main__":
    main()
