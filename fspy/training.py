"""Implements functions for training and testing a model."""
import datetime

from fspy.datasets import PredictionDataset

import torch
from torch.utils.data import DataLoader

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCH_SIZE = 64
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# pylint: disable=too-many-arguments, too-many-locals
def train(model,
          dataset,
          batch_size=DEFAULT_BATCH_SIZE,
          num_epochs=DEFAULT_EPOCH_SIZE,
          loss_func=torch.nn.BCELoss(),
          device=DEFAULT_DEVICE,
          quiet=False):
    """Trains a model over a dataset.

    The model is trained using the SGD optimizer with a learning rate of 0.01.

    Arguments:
        model (torch.nn.Module): A PyTorch model.
        dataset (list): A list of (x, y) pairs.
        batch_size (int, optional): The training batch size.
        num_epochs (int, optional): The number of epochs to propogate over.
        loss_func (callable, optional): The cost function.
        device (str): The device on which to run the test.
        quiet (bool, optional): If true, then this function will not print
            status messages to STDOUT.
    """

    def print_train_settings(dataset, batch_size, num_epochs, device):
        """Prints information about the training hyperparameters."""
        assert dataset, "Expected non-empty dataset."

        start_time = datetime.datetime.now()
        print("Starting training at time %s." % start_time, end="\n\n")

        print("BATCH_SIZE=%d" % batch_size)
        print("NUM_EPOCHS=%d" % num_epochs)
        if torch.cuda.device_count() > 1:
            print("NUM_GPUS=%d" % torch.cuda.device_count())
        print("DEVICE=%s" % device, end="\n\n")

        features, label = dataset[0]
        print("INPUT_SHAPE=", features.shape, sep="")
        print("LABEL_SHAPE=", label.shape, sep="")
        print("DATASET_SIZE=%d" % len(dataset), end="\n\n")

    def print_batch_results(epoch, batch, loss):
        if epoch == 0 and batch == 0:
            print("EPOCH\tBATCH\tLOSS")
        print("%d\t%d\t%.4f" % (epoch, batch, loss.item()))

    dataloader = DataLoader(PredictionDataset(dataset),
                            batch_size,
                            shuffle=True)
    if not quiet:
        print_train_settings(dataset, batch_size, num_epochs, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    if torch.cuda.device_count() > 1 and not device == "cpu":
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch_number, (observations, labels) in enumerate(dataloader):
            model.zero_grad()

            observations = observations.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            observations = observations.to(device)
            labels = labels.to(device)

            predictions = model(observations)
            # torch.Size([N, 1]) => torch.Size([N])
            predictions = torch.squeeze(predictions)
            loss = loss_func(predictions, labels)

            if not quiet:
                print_batch_results(epoch, batch_number, loss)

            loss.backward()
            optimizer.step()

    return model


def test(model,
         dataset,
         batch_size=DEFAULT_BATCH_SIZE * 2,
         device=DEFAULT_DEVICE,
         quiet=False):
    """Tests a model on the specified dataset.

    This function assumes that predictions lay in the range [0, 1].
    Predictions greater than 0.5 are classified as 1, and 0 otherwise.

    Arguments:
        model (torch.nn.Module): A PyTorch model.
        dataset (list): A list of (x, y) pairs.
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

    dataloader = DataLoader(PredictionDataset(dataset), batch_size)
    batch_accuracies = []

    model.eval()
    with torch.no_grad():
        for observations, labels in dataloader:
            observations = observations.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            observations = observations.to(device)
            labels = labels.to(device)

            predictions = model(observations)
            batch_accuracy = average_accuracy(predictions, labels)

            batch_accuracies.append(batch_accuracy)

    average_batch_accuracy = sum(batch_accuracies) / len(batch_accuracies)

    if not quiet:
        print_test_results(average_batch_accuracy)

    return average_batch_accuracy


def save(model, filename, quiet=False):
    """Saves a model's state dictionary to a file.

    Arguments:
        model (torch.nn.Module): The PyTorch model to save.
        filename (string): Desired model filename.
        quiet (bool, optional): If true, then this function will not print
            status messages to STDOUT.
    """
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), filename)
    else:
        torch.save(model.state_dict(), filename)
    if not quiet:
        print("Saved model to %s" % filename)
