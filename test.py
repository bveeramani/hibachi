import pandas

dataframe = pandas.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")
X = torch.tensor(dataframe.values[:, 0:8])
y = torch.tensor(dataframe.values[:, 8])

from hibachi.filter import filters, criteria
from hibachi.wrapper import wrappers, models, accuracy
filters.VarianceThreshold(minimum=0.5)
filters.Missing(maximum=0.2)


filter1 = filters.Threshold(cutoff=0.5, maximize=True, criterion=criteria.Variance())
filter2 = filters.Select(k=3, maximize=True, criterion=criteria.CCM(m=3))
filter3 = filters.Custom(lambda score: 0 < score < 1, criterion=criteria.Variance())
select = filters.Compose(filter1, filter2, filter3)
select = wrappers.SFFS(model_class=models.SVM, regression=False)

from hibachi.functional import algorithms, objectives

from hibachi import objectives

select = filters.Constant()
select = filters.Select(k=3, criterion=criteria.CCM(m=3))
X = select(X, y)

select = filters.Compose(
    filters.CorrelationThreshold(cutoff=0.9),
    filters.VarianceThreshold(cutoff=0.1),
    filters.MissingThreshold(cutoff=0.6),
    filters.CollinearityThreshold(cutoff=0.9),
)
X = select(X, y)

select = wrappers.SFFS(model_class=models.SVM)
X = select(X, y)


objectives.

algorithms.SFFS(objective=objectives.NegativeMSE(model_class, train))


filter1(X, y)
from hibachi.interpretation import explainers

explainer = explainers.DeepSHAP(asf, asdfasf)
attributions = explainer(model, image)

from hibachi import visualization
visualization.heatmap(attributions)

criterion = criteria.Variance()
weights = criterion(X, y)

select(X, k=3, scores)
