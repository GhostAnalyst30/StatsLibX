from statslibx import load_dataset, DescriptiveStats, InferentialStats
import pandas as pd
# df = pd.read_csv(r"tests\bank (1).csv", sep=";")

# df = load_dataset(r"tests\bank (1).csv", sep=";")
# stats = DescriptiveStats(df)
# print(stats.data)

# infer = InferentialStats(df)
# print(infer.data)

# df = load_dataset(r"statslibx\datasets\Cocoa_Bubbles_Investment_Nigeria_Ghana_1980_2023.xlsx")

# ds = DescriptiveStats(df)

# print(ds.data)

import statslibx as slx

df = slx.datasets.load_penguins()
infer = slx.InferentialStats(df)

# Confidence Interval + Point Estimate
print(infer.confidence_interval(
    column="bill_length_mm",
    statistic="mean"
))



