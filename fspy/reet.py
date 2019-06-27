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
    print("yeeting", num_features)
    selection_algorithm = lambda dataset: pearson_select(dataset, num_features)
    stability = measure_stability(selection_algorithm,
                                  dataset,
                                  stability_measure=tanimoto_distance,
                                  num_folds=2)
    selection_stabilities.append(stability)
print(selection_stabilities)

accuracies = []
for num_features in range(1, 25):
    print("reeting", num_features)
    feature_subset = pearson_select(dataset, num_features)
    accuracy = measure_accuracy(feature_subset, dataset, num_folds=5)
    accuracies.append(accuracy)
print(accuracies)
