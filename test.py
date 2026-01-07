from statslibx import DescriptiveStats, InferentialStats, UtilsStats
from statslibx.datasets import load_dataset, load_iris, load_penguins

data = load_penguins()

descriptive = DescriptiveStats(data)

print(descriptive.data.head())