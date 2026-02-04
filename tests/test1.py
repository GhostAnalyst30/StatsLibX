from statslibx import load_dataset, DescriptiveStats, InferentialStats
import pandas as pd
# df = pd.read_csv(r"tests\bank (1).csv", sep=";")

# df = load_dataset(r"tests\bank (1).csv", sep=";")
# stats = DescriptiveStats(df)
# print(stats.data)

# infer = InferentialStats(df)
# print(infer.data)

df = load_dataset(r"C:\Users\Usuario\Documents\Emmanuel Ascendra\Ciencia de Datos\Librerias Python\__huggingface_repos__.json")

ds = DescriptiveStats(df)

print(ds.data)




