from src import train_model
import numpy as np
from sklearn.metrics import f1_score


# hyperparameter search to find best params for given models and datasets

rand_seed = 1234
do_log_wandb = False
batch_size = 4
n_epochs = 10

for dataset in ["zurich", "newyork", "tokyo", "all"]:

	for model_name in [ 
		{ "full": "facebook/convnext-base-224-22k-1k", "short": "cnbase" }, 
		{ "full": "google/vit-large-patch16-224", "short": "vitlarge" },
		{ "full": "microsoft/swinv2-large-patch4-window12-192-22k", "short": "swinv2" },
	]:

		run_id = f'2023-{model_name.get("full")}-{dataset}'

		dataset_name = "facadematerials-" + dataset

		m_short = model_name.get("short")
		filename = f"results-{dataset}-{m_short}.csv"

		with open(filename, "a") as fp:
			header = [ "lr", "gradient_acc_steps", "weight_decay", "randaugm_m", "score"]
			fp.write(','.join(header) + "\n")

		for learning_rate in [ 1e-4, 5e-5, 2e-5 ]:
			for gradient_acc_steps in [ 4, 8, 16 ]:
				for weight_decay in [ 0.01, 0.05, 0.1, None ]:
					for randaugm_m in [ 8, 16 ]:

						train_model(model_name.get("full"), dataset_name, n_epochs, run_id, learning_rate, batch_size, gradient_acc_steps, weight_decay, randaugm_m, rand_seed, do_log_wandb, do_only_first_batch=False, filename=filename)