from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

cmap = cm.get_cmap('viridis', 5)
csfont = {'fontname':'Times New Roman'}

train_samples = [
	[ 69, 56, 53, 75, 98, 54 ],
	[ 52, 41, 53, 41, 63, 39 ],
	[ 76, 60, 67, 22, 74, 65 ],
	[ 195, 43, 161, 157, 97, 71, 234, 95 ]
]

f1_mean = [
	[ 0.98666667, 0.9454191, 0.95475113, 0.95054945, 0.89650794, 0.93732194 ],
	[ 0.84867725, 0.92592593, 1.0, 0.9280303, 0.95238095, 0.77128427 ],
	[ 0.92668372, 0.91880342, 0.85978836, 1.0, 0.87499031, 0.90909091 ],
	[ 0.94227813, 0.9047619, 0.95919493, 0.89504608, 0.96310892, 0.95008913, 0.93575531, 0.88201058 ],
]

f1_std = [
	[ 0.01885618, 0.04538186, 0.03283792, 0.03503869, 0.04059522, 0.04646583 ],
	[ 0.03677915, 0.05237828, 0.0, 0.05275905, 0.0673435, 0.14466404 ],
	[ 0.07506974, 0.06810845, 0.02275747, 0.0, 0.1046398, 0.07422696 ],
	[ 0.02945242, 0.13468701, 0.04093713, 0.05600572, 0.00796044, 0.03764476, 0.01083344, 0.01815979 ]
]

labels = ["Zurich (ZRH)", "Tokyo (TKY)", "New York City (NYC)", "All"]

for idx, dataset in enumerate(["zurich", "tokyo", "newyork", "all"]):

	dataset_name = "facadematerials_" + dataset + "-1"

	df = pd.read_csv(f"{dataset_name}/train/_classes.csv")
	facade_labels = list(df.columns)[1:]

	n_samples = train_samples[idx]

	xs = np.arange(len(n_samples))

	#print([(sample / np.max(n_samples)) for sample in n_samples])
	#print([(210, 210, 210, sample / np.max(n_samples)) for sample in n_samples])

	fig,ax = plt.subplots()
	print(cmap.colors[1])
	ax.bar(xs, n_samples, color=[(cmap.colors[0][0], cmap.colors[0][1], cmap.colors[0][2], (sample / np.max(n_samples))**0.1 - 0.75) for sample in n_samples])
	ax.set_xticks(xs, facade_labels, rotation=90, fontsize=16, **csfont)
	ax.set_ylabel("Samples in Training Set", fontsize=16, **csfont)

	ax2=ax.twinx()
	ax2.errorbar(xs, f1_mean[idx], f1_std[idx], linestyle='None', marker='D', color=cmap.colors[1], mec=cmap.colors[1], mfc=cmap.colors[1], markersize=8)
	#ax2.fill_between(xs, np.subtract(f1_mean[idx], f1_std[idx]), np.add(f1_mean[idx], f1_std[idx]), alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=4, linestyle='dashdot', antialiased=True)

	ax2.set_ylim(0.5, 1.09)
	ax2.set_ylabel("F1 Score", fontsize=16, **csfont)
	
	plt.title(f"Class-Wise F1 Scores for {labels[idx]}", fontsize=16, **csfont)
	plt.savefig(f"f1-scores-{dataset}-hist.svg")

	plt.show()
