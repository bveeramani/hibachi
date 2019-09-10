# Hibachi
**Hibachi** is a Python package that implements feature selection and model
interpretation methods for PyTorch.

## Installation
Run the following command:

```
pip install hibachi
```

## Roadmap

### 0.1.0 (Release 9/11/2019)
#### hibachi.filter.filters
* `CollinearityThreshold`

#### hibachi.filter.criteria
* `Collinearity`

#### hibachi.interpretation.visualization
* `heatmap`

#### hibachi.algorithms
* `SFS`
* `SBS`
* `SFFS`
* `SFBS`
* `Greedy`
* `SuperGreedy`

#### hibachi.models
* `Linear`
* `Logistic`

#### hibachi.objectives
* `NegativeMSE`
* `ClassificationAccuracy`

#### hibachi.wrappers
* `SFS`
* `SBS`
* `SFFS`
* `SFBS`
* `Greedy`
* `SuperGreedy`

#### hibachi.interpretation.explainers
* `GradientStarInput`
* `Occulsion`
* `Saliency`

### 0.2.0 (Release 9/13/2019)
#### hibachi.filter.filters
* `MutualInformationThreshold`
* `ChiSquareThreshold`

#### hibachi.filter.criteria
* `MutualInformation`
* `ChiSquare`

#### hibachi.interpretation.explainers
* `IntegratedGradient`
* `DeepLIFT`
* `LIME`

#### hibachi.wrappers
* `LasVegas`
* `LasVegasIncremental`
* `QBB`

#### hibachi.algorithms
* `LasVegas`
* `LasVegasIncremental`
* `QBB`

#### hibachi.models
* `SVM`

### Development
In-progress, timelines long or uncertain

#### hibachi.filter.filters
* `RandomForest`

#### hibachi.filter.criteria
* `RandomForestWeights`
* `ANOVA`

#### hibachi.wrappers
* `Genetic`
* `RELIEF`

#### hibachi.algorithms
* `Genetic`
* `RELIEF`

#### hibachi.objectives
* `ChernoffDivergence`
* `BhattacharyyaDivergence`
* `KLDivergence`
* `KolmogorovDivergence`
* `MatusitaDivergence`
* `PatrickFisherDivergence`
* `Dependence`
* `Distance`
* `Uncertainty`
* `Consistency`

#### hibachi.interpretation.explainers
* `LRP`
* `EpsilonLRP`
* `SHAP`
* `KernelSHAP`
* `MaxSHAP`
* `DeepSHAP`
* `LowOrderSHAP`
* `L2X`
