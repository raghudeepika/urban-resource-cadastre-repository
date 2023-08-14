from src import train_model
import numpy as np
from sklearn.metrics import f1_score


#model_name = "facebook/convnext-base-224-22k-1k"
#model_name = "google/vit-large-patch16-224"
model_name = "microsoft/swinv2-large-patch4-window12-192-22k"

n_epochs = 10
batch_size = 4

run_id = "2023-04-24-all"

dataset_name = "facadematerials-zurich"

learning_rate = 1e-5
gradient_acc_steps = 8
weight_decay = 0.1
randaugm_m = 16

do_log_wandb = False

results = []
classwise_f1_scores = []

#pretrained_model_pth = "./model-swinv2-all.pt"
pretrained_model_pth = None


for rand_seed in [ 1234, 3456, 5678 ]:
	emr, hd, prec, rec, f1, y_true, y_pred = train_model(model_name, dataset_name, n_epochs, run_id, learning_rate, batch_size, gradient_acc_steps, weight_decay, randaugm_m, rand_seed, do_log_wandb, save_results=True, pretrained_model_pth=pretrained_model_pth)
	classwise_f1_scores.append([ f1_score(y_true[:, idx], y_pred[:, idx]) for idx in range(y_true.shape[1]) ])
	results.append([ emr, hd, prec, rec, f1 ])

results_mean = np.mean(results, axis=0)
result_std = np.std(results, axis=0)

print("emr:", round(results_mean[0], 2), round(result_std[0], 2))
print("hd:", round(results_mean[1], 2), round(result_std[1], 2))
print("prec:", round(results_mean[2], 2), round(result_std[2], 2))
print("rec:", round(results_mean[3], 2), round(result_std[3], 2))
print("f1 (macro):", round(results_mean[4], 2), round(result_std[4], 2))

print("f1 classwise (mean):", np.mean(classwise_f1_scores, axis=0))
print("f1 classwise (std):", np.std(classwise_f1_scores, axis=0))