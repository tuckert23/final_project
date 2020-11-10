import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap

df = pd.read_csv("./cleaned_data_final.csv")

columns = df.columns

# Scatterplot Matrix
scatterplotmatrix(df[columns].values, figsize=(30, 30), names=columns, alpha=0.8)
plt.tight_layout()
plt.savefig("./figs/scatterplot_matrix.png")

# Correlation heatmap
corrmap = np.corrcoef(df[columns].values.T)
hmap = heatmap(corrmap, row_names=columns, column_names=columns, figsize=(20, 20))
plt.savefig("./figs/heatmap.png")

