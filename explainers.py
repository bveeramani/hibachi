"""Implements various model attribution methods"""
from string import ascii_uppercase

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import torch

import ccm


def CCMExplainer(num_features, epsilon=0.01, target_type='binary'):
    """Returns an attribution method that uses the Conditional Covariance
    Minimization algorithm.

    Arguments:
        num_features (optional, int): The number of features to select.
        epsilon (optional, float): The regularization parameter.
        target_type (optional, string): This argument is currently not supported.

    Returns:
        A function that takes in a dataset and returns the selected features.
    """
    assert target_type == "binary", "Non-binary target types are not supported"

    def explain(dataset):
        """Prints CCM attribution for a dataset."""
        # An N x K array where N is the number of samples and K is the number of features
        feature_matrix = np.array([list(features) for features, label in dataset])
        # An length-N 1-dimensional array where N is the number of samples
        label_matrix = np.array([1 if label == 1 else -1 for features, label in dataset])
        rank = ccm.ccm(feature_matrix, label_matrix, num_features, target_type, epsilon, iterations=100, verbose=False)
        selected_features = np.argsort(rank)[:num_features]

        print('The features selected by CCM are features {}'.format(selected_features))
        return selected_features

    return explain


def LimeExplainer(model, dataset, kwargs):
    """Returns a tabular Lime attribution method.

    Arguments:
        model (callable): A PyTorch model.
        dataset (torch.utils.data.Dataset): The training dataset.
        kwargs (optional, dictionary): Dictionary of keyword arguments that
            should be used to construct the Lime explainer.

    Returns:
        A function that accepts a dataset sample as input and displays the
        local explanation for that sample. Each sample should be a two-tuple
        where the first element contains the features and the second element
        contains the label.
    """

    def wrap(model):
        """Wraps the model function so that the model accepts a 2-dimensional
        feature array as input and returns a 1-dimensional probability array.

       Wrapping the model allows us to use the model as an argument in
       Lime explainers.
       """

        def wrapped_model(input):
            model.eval()
            with torch.no_grad():
                input = torch.tensor(input).type(torch.FloatTensor)
                predictions = model(input)
                probabilities = np.array(predictions).squeeze()
                complementary_probabilities = 1 - probabilities
                columns = complementary_probabilities, probabilities
                return np.column_stack(columns)

        return wrapped_model

    model = wrap(model)
    feature_list = [list(features) for features, label in dataset]
    # An N x K array where N is the number of samples and K is the number of features
    feature_matrix = np.array(feature_list)
    explainer = LimeTabularExplainer(feature_matrix, **kwargs)

    def explain(sample):
        """Plots Lime attribution for a feature-label pair."""
        features = sample[0]
        features = np.array(features)

        explanation = explainer.explain_instance(features, model)
        figure = explanation.as_pyplot_figure()
        figure.show()

        return explanation

    return explain


def OteyP450LimeExplainer(model, dataset):
    """Returns an attribution method for the OteyP450 dataset that uses the Lime algorithm."""
    kwargs = {}
    kwargs["feature_names"] = (i + 1 for i in range(8))
    kwargs["class_names"] = ("non-functional", "functional")
    kwargs["categorical_names"] = {
        i: {j + 1: ascii_uppercase[j] for j in range(3)} for i in range(8)
    }
    kwargs["categorical_features"] = (i for i in range(8))
    return LimeExplainer(model, dataset, kwargs)
