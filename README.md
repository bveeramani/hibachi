# Hibachi
**Hibachi** is a Python package that implements feature selection methods for
PyTorch.

## Installation
Run the following command:

```
pip install hibachi
```

## To-do
| Feature Name                           | Feature Selector API                                        | sklearn API                                             |
|----------------------------------------|-------------------------------------------------------------|---------------------------------------------------------|
| Missing values                         | `fs.identify_missing(missing_threshold=0.6)`                |                                                         |
| Single Unique Value                    | `fs.identify_single_unique()`                               |                                                         |
| Collinear (highly correlated) Features | `fs.identify_collinear(correlation_threshold=0.98)`         |                                                         |
| Low variance                           |                                                             | `VarianceThreshold(threshold=(.8 * (1 - .8)))`          |
| Chi, F-value, mutual info              |                                                             | `SelectKBest(chi2, k=2).fit_transform(X, y)`            |
| Recursive feature elimination          |                                                             | `RFE(estimator=svc, n_features_to_select=1, step=1)`    |
| L1-based feature selection             |                                                             | `LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)` |
| Tree-based feature selection           | `fs.identify_zero_importance(task = 'classification')`      | `SelectFromModel(clf, prefit=True)`                     |
| Tree-based feature selection           | `fs.identify_low_importance(cumulative_importance = 0.99)`  | `SelectFromModel(clf, prefit=True)`                     |


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
