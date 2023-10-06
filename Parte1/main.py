from datasets import load_dataset
import numpy as np

dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]
numpyEdades = np.array(data['age'])
promEdades = numpyEdades.mean()