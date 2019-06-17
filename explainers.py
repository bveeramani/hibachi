"""Implements various model attribution methods"""
from string import ascii_uppercase

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import torch


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
