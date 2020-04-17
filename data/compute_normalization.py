import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)

args = parser.parse_args()

data = pd.read_pickle(args.file)

print("File:", args.file)
print("\tMean:", np.mean(data["YY"]))
print("\tStd:", np.std(data["YY"]))
print("\tMin:", np.min(data["YY"]))
print("\tMax:", np.max(data["YY"]))
print("\tCount:", len(data))
print()

print(data.head())
