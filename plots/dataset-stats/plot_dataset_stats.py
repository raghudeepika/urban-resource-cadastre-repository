from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from roboflow import Roboflow
from matplotlib import cm

cmap = cm.get_cmap('viridis', 8)
csfont = {'fontname':'Times New Roman'}

labels = ["Zurich (ZRH)", "New York City (NYC)", "Tokyo (TKY)", "All"]

for idx, dataset in enumerate([ "zurich", "newyork", "tokyo", "all" ]):

	dataset_name = "facadematerials_" + dataset

	rf = Roboflow(api_key="cDQPey9gVV6b28l6rMcm")
	project = rf.workspace("rcr").project(dataset_name)
	ds = project.version(1).download("multiclass")
	ds_root = ds.location

	# ds_root = "facadematerials_zurich-1"

	df = pd.read_csv(f"{ds_root}/train/_classes.csv")
	all_data = np.asarray(df)

	n_classes = all_data[:, 1:].shape[1]

	facade_labels = list(df.columns)[1:]

	y_all = all_data[:, 1:]

	plt.bar(range(y_all.shape[1]), np.count_nonzero(y_all, axis=0), color=cmap.colors)
	plt.xticks(range(y_all.shape[1]), facade_labels, rotation=90, fontsize=16, **csfont)
	plt.yticks(fontsize=16, **csfont)

	plt.title(f"Class Distribution for {labels[idx]}", fontsize=16, **csfont)

	#plt.savefig("dataset-zurich-hist.svg")
	#plt.savefig("dataset-newyork-hist.svg")
	#plt.savefig("dataset-tokyo-hist.svg")
	plt.savefig(f"dataset-{dataset}-hist.svg")

	#plt.show()