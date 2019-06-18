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
  --batches BATCH_SIZE  number of samples to propogate (default: 64)
  --epochs NUM_EPOCHS   number of passes through dataset (default: 32)
  --save FILENAME       save trained model (default "model.py")
  --disable-cuda        disable CUDA support
"""
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import OteyP450
from models import SupportVectorMachine

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCH_SIZE = 64
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_SAVE_FILENAME = "model.pt"

ModelType = SupportVectorMachine
transform = lambda features, label : (features, torch.tensor(1)) if label else (features, torch.tensor(-1))
MODEL_ARGS = [OteyP450("OteyP450/processed/train.csv", transform=transform)]
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

    transform = lambda features, label : (features, torch.tensor(1)) if label else (features, torch.tensor(-1))
    train_dataset = OteyP450(args.train_dataset, transform=transform)
    test_dataset = OteyP450(args.test_dataset, transform=transform) if args.test_dataset else None

    model = ModelType(*MODEL_ARGS, **MODEL_KWARGS)
    if torch.cuda.device_count() > 1 and not args.cuda_disabled:
        model = nn.DataParallel(model)
    model = model.to(device)

    model = train(model,
                  train_dataset,
                  batch_size=args.batch_size,
                  num_epochs=args.num_epochs,
                  loss_func=torch.nn.HingeEmbeddingLoss(),
                  device=device,
                  test_dataset=test_dataset)

    if args.test_dataset:
        test(model,
             test_dataset,
             loss_func=torch.nn.HingeEmbeddingLoss(),
             batch_size=args.batch_size * 2,
             device=device)

    if args.save_filename:
        save(model, args.save_filename)

    return 0


#pylint: disable=too-many-arguments, too-many-locals
def train(model,
          train_dataset,
          batch_size=DEFAULT_BATCH_SIZE,
          num_epochs=DEFAULT_EPOCH_SIZE,
          loss_func=nn.BCELoss(),
          device=DEFAULT_DEVICE,
          test_dataset=None):
    """Trains the specified model.

    The specified model will always be trained using the SGD optimizer.
    """

    def debug_train(train_dataset, batch_size, num_epochs, device):
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


    dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    debug_train(train_dataset, batch_size, num_epochs, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        model.train()
        for batch_number, (features, labels) in enumerate(dataloader):
            model.zero_grad()

            features = features.to(device)
            labels = labels.to(device)

            predictions = model(features)
            loss = loss_func(predictions, labels)

            x = loss.backward()
            print(x)
            optimizer.step()

    return model


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
        if prediction > 0 and label == 1 or prediction <= 0 and label == -1:
            num_correct += 1

    return num_correct / len(predictions)


def test(model,
         dataset,
         batch_size=DEFAULT_BATCH_SIZE * 2,
         accuracy_func=average_accuracy,
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
            print(predictions, labels)
            batch_accuracy = accuracy_func(predictions, labels)

            total_loss += loss_func(predictions, labels)
            batch_accuracies.append(batch_accuracy)

    average_batch_accuracy = sum(batch_accuracies) / len(batch_accuracies)
    debug_test(total_loss, average_batch_accuracy)


def save(model, filename):
    """Saves a model's state dictionary to a file.

    Arguments:
        model (nn.Module): The PyTorch model to save.
        filename (string): Desired model filename.
    """
    print("Saving model to %s" % filename)
    torch.save(model.module.state_dict(), filename)


if __name__ == "__main__":
    main()
