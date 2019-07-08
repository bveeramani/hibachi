import numpy as np

from fspy import datasets
from fspy import algorithms

print("Generating orange skin dataset.")
dataset = datasets.OrangeSkin(500)

for e_n in [pow(10, x) for x in range(-5, 2)]:
    print("Running experiment with se = %d." % e_n)
    medians = []
    for trial in range(10):
        print("Running trial %d." % trial)
        ranking = algorithms.ccm_rank(dataset, 4, .001)
        median = np.median(ranking[0:4])
        medians.append(median)
    print("Results:")
    print("\tmean =", np.mean(medians))
    print("\tmax =", np.max(medians))
    print("\tmin =", np.min(medians))
    print("\n")
