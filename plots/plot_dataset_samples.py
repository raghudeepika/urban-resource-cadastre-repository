from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from roboflow import Roboflow
from matplotlib import cm
from PIL import Image


n_imgs = 5

f, axs = plt.subplots(3, n_imgs)

for idx, dataset in enumerate(["facadematerials_zurich-1", "facadematerials_tokyo-1", "facadematerials_newyork-1"]):

	df = pd.read_csv(f"{dataset}/train/_classes.csv")
	facade_labels = list(df.columns)[1:]
	all_data = np.asarray(df)
	x_pths = all_data[:, 0]
	
	sample_idxs = np.random.choice(x_pths.shape[0], n_imgs)
	sample_pths = x_pths[sample_idxs]
	sample_labels = all_data[:, 1:][sample_idxs]

	print(sample_labels[0])
	print(sample_labels[0].nonzero()[0].astype(np.int32))

	for idx_pth, pth in enumerate(sample_pths):

		img = Image.open(dataset + "/train/" + pth)

		ys = sample_labels[idx_pth].nonzero()[0].astype(np.int32)
		title = ' +'.join([ facade_labels[y] for y in ys ])

		axs[idx, idx_pth].imshow(img)
		axs[idx, idx_pth].set_axis_off()
		axs[idx, idx_pth].set_title(title, size=16)

		img.close()

plt.tight_layout()
plt.show()

f.savefig("dataset_samples.svg")