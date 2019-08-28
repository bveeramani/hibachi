from hibachi import datasets

dataset = datasets.OrangeSkin(n=100)

from hibachi import wrappers

select = wrappers.SuperGreedy()
subset = select(dataset, k=4)
print(subset)
