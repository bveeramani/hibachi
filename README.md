# FSPY
**FSPY** (**F**eature **S**election **Py**thon) is a Python package that
provides routines for running and comparing various feature selection algorithms.

## Example
```
from fspy.algorithms import pearson_rank, pearson_select
from fspy.datasets import OteyP450
from fspy.measurements import measure_stability, measure_accuracy, spearman_coefficient, tanimoto_distance

dataset = OteyP450("data/OteyP450/raw/otey_P450_binary_function.txt")

rank_stability = measure_stability(pearson_rank,
                                   dataset,
                                   stability_measure=spearman_coefficient,
                                   num_folds=5)
print(rank_stability)

selection_stabilities = []
for num_features in range(1, 25):
    selection_algorithm = lambda dataset: pearson_select(dataset, num_features)
    stability = measure_stability(selection_algorithm,
                                  dataset,
                                  stability_measure=tanimoto_distance,
                                  num_folds=2)
    selection_stabilities.append(stability)
print(selection_stabilities)

accuracies = []
for num_features in range(1, 25):
    feature_subset = pearson_select(dataset, num_features)
    accuracy = measure_accuracy(feature_subset, dataset, num_folds=5)
    accuracies.append(accuracy)
print(accuracies)
```

## References
```
@incollection {
    NIPS2017_7270,
    title = {Kernel Feature Selection via Conditional Covariance Minimization},
    author = {Chen, Jianbo and Stern, Mitchell and Wainwright, Martin J and
        Jordan, Michael I},
    booktitle = {Advances in Neural Information Processing Systems 30},
    editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R.
        Fergus and S. Vishwanathan and R. Garnett},
    pages = {6949--6958},
    year = {2017},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/7270-kernel-feature-selection-via-conditional-covariance-minimization.pdf}
}
```
