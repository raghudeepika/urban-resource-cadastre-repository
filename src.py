from PIL import Image
import glob
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, classification_report, precision_score
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
from transformers import ViTModel, ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor, EfficientNetImageProcessor, EfficientNetForImageClassification, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, TrainingArguments, pipeline, AutoFeatureExtractor, AutoModelForImageClassification, AutoImageProcessor
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor, ConvertImageDtype, RandomCrop
import numpy as np
import random
import os
import sklearn
import torchvision.transforms as T
import wandb
import pdb
from RandAugment import RandAugment
import torch.nn.functional as F
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from matplotlib import pyplot as plt
from roboflow import Roboflow


def get_tgseed(seed):
	g = torch.Generator()
	g.manual_seed(seed)
	return g


def set_seeds(seed):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	torch.cuda.manual_seed(seed)
	torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.deterministic = True

def get_key_from_dict_idx(labels, idx):
	return list(labels.keys())[idx]


def log_metric_on_wandb(metric, val, epoch, do_log_wandb):
	if do_log_wandb: wandb.log({metric: val, "epoch": epoch})


def multihot_to_label(multihot_tensor):
	idxs = multihot_tensor.detach().cpu().squeeze().nonzero().numpy()[:, 0].tolist()
	labels = [ facade_labels[idx] for idx in idxs ]
	return labels


def rand_bbox(size, lam):
	W = size[2]
	H = size[3]
	cut_rat = np.sqrt(1. - lam)
	cut_w = np.int32(W * cut_rat)
	cut_h = np.int32(H * cut_rat)

	# uniform
	cx = np.random.randint(W)
	cy = np.random.randint(H)

	bbx1 = np.clip(cx - cut_w // 2, 0, W)
	bby1 = np.clip(cy - cut_h // 2, 0, H)
	bbx2 = np.clip(cx + cut_w // 2, 0, W)
	bby2 = np.clip(cy + cut_h // 2, 0, H)

	return bbx1, bby1, bbx2, bby2


def compute_metrics(y_true, y_pred, loss_all, dataset_split, epoch, labels_classes, do_log_wandb=True):

	score_emr = np.all(y_pred == y_true, axis=1).mean()

	score_hamming = sklearn.metrics.hamming_loss(y_true, y_pred)
	
	score_f1 = np.mean([ f1_score(y_true[:, idx], y_pred[:, idx]) for idx in range(y_true.shape[1]) ])
	#score_f1_samples = f1_score(y_true, y_pred, average='samples')
	
	score_recall = np.mean([ recall_score(y_true[:, idx], y_pred[:, idx]) for idx in range(y_true.shape[1]) ])
	score_prec = np.mean([ precision_score(y_true[:, idx], y_pred[:, idx]) for idx in range(y_true.shape[1]) ])

	score_f1_class_first = f1_score(y_true[:, 0], y_pred[:, 0])
	score_f1_class_last = f1_score(y_true[:, -1], y_pred[:, -1])

	print("")
	print(f"[ {dataset_split} ] exact match ratio (EMR): ", score_emr)
	print(f"[ {dataset_split} ] hamming loss: ", score_hamming)
	print(f"[ {dataset_split} ] precision (macro)", score_prec)
	print(f"[ {dataset_split} ] recall (macro)", score_recall)
	print(f"[ {dataset_split} ] f1 score (macro): ", score_f1)
	
	print(f"[ {dataset_split} ] f1 score for class '{labels_classes[0]}'", score_f1_class_first)
	print(f"[ {dataset_split} ] f1 score for class '{labels_classes[-1]}'", score_f1_class_last)
	
	print("")

	if do_log_wandb:
		log_metric_on_wandb(f"[ {dataset_split} ] loss", np.mean(loss_all), epoch, do_log_wandb)
		
		log_metric_on_wandb(f"[ {dataset_split} ] exact match ratio (EMR): ", score_emr, epoch, do_log_wandb)
		log_metric_on_wandb(f"[ {dataset_split} ] hamming loss", score_hamming, epoch, do_log_wandb)
		log_metric_on_wandb(f"[ {dataset_split} ] f1 score (macro)", score_f1, epoch, do_log_wandb)
		log_metric_on_wandb(f"[ {dataset_split} ] recall", score_recall, epoch, do_log_wandb)
		log_metric_on_wandb(f"[ {dataset_split} ] recall (macro)", score_recall, epoch, do_log_wandb)
		log_metric_on_wandb(f"[ {dataset_split} ] precision (macro)", score_prec, epoch, do_log_wandb)

		log_metric_on_wandb(f"[ {dataset_split} ] f1 score for class '{labels_classes[0]}'", score_f1_class_first, epoch, do_log_wandb)
		log_metric_on_wandb(f"[ {dataset_split} ] f1 score for class '{labels_classes[-1]}'", score_f1_class_last, epoch, do_log_wandb)

	return score_emr, score_hamming, score_prec, score_recall, score_f1
	

class BuildingFacadeDataset(Dataset):

	def __init__(self, data, width, height, img_root, mode, image_processor, randaugm_n=None, randaugm_m=None, do_augm=False):
		
		self.x_pths = data[:, 0].tolist()
		self.y = data[:, 1:].tolist()
		
		self.mode = mode
		self.do_augm = do_augm
		self.img_root = img_root
		self.width = width
		self.height = height
		
		self.randaugm_n = randaugm_n
		self.randaugm_m = randaugm_m

		self.fn_normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

		inv_norm1 = Normalize(mean=[0.,0.,0.], std=(torch.divide(torch.tensor([1.0]), torch.tensor(image_processor.image_std))))
		inv_norm2 = Normalize(mean=torch.sub(torch.tensor([0.0]), torch.tensor(image_processor.image_mean)), std=[1., 1., 1.])
		self.fn_normalize_inv = Compose([ inv_norm1 , inv_norm2 ])
		
		self.model_height = image_processor.size.get("height") if image_processor.size.get("height") is not None else image_processor.size.get("shortest_edge")
		self.model_width = image_processor.size.get("width") if image_processor.size.get("width") is not None else image_processor.size.get("shortest_edge")
		
		print("image size for model:", self.model_height, self.model_width)

		if self.mode == "train":
			self.fn_transform = Compose(
				[
					ToTensor(),
					RandomCrop([np.min([self.width, self.height]), np.min([self.width, self.height])]),
					Resize((self.model_height, self.model_width), antialias=True),
					RandomHorizontalFlip(),
					ConvertImageDtype(torch.float),
					#self.fn_normalize,
				]
			)
			if self.do_augm:
				rand_augm = RandAugment(self.randaugm_n, self.randaugm_m)
				rand_augm.augment_list.pop(0) # autocontrast
				rand_augm.augment_list.pop(1) # invert
				rand_augm.augment_list.pop(2) # posterize
				rand_augm.augment_list.pop(2) # solarize
				rand_augm.augment_list.pop(2) # solarizeadd
				self.fn_transform.transforms.insert(0, rand_augm)
		else:
			self.fn_transform = Compose(
				[
					ToTensor(),
					RandomCrop([np.min([self.width, self.height]), np.min([self.width, self.height])]),
					Resize((self.model_height, self.model_width), antialias=True),
					ConvertImageDtype(torch.float),
					self.fn_normalize,
				]
			)

	def __len__(self):
		return len(self.x_pths)

	def __getitem__(self, idx):
	
		img = Image.open(self.img_root + "/" + self.x_pths[idx])
		x = img.copy()
		
		y = torch.tensor(self.y[idx], dtype=torch.float32)

		x = self.fn_transform(x)
		
		# x = np.transpose(x.cpu().numpy(), (1, 2, 0))
		# plt.tight_layout()
		# plt.axis('off')
		# plt.savefig(f"/Users/mnbucher/Downloads/paper-new-{np.random.randint(0, 100)}.png", bbox_inches='tight')
		# plt.show()
		# exit()

		img.close()
		
		return x, self.x_pths[idx], y


def do_epoch(model, dataloader, dataset, dataset_split, n_classes, gradient_acc_steps, optimizer, scheduler, loss_fn, epoch, labels_classes, dvc, do_log_wandb):

	y_true_all = np.zeros((len(dataset), n_classes))
	y_pred_all = np.zeros((len(dataset), n_classes))
	x_pths = []
	loss_all = np.zeros(len(dataloader))

	idx = 0
	optimizer.zero_grad()

	for idx_batch, (x, x_pth, y_true) in enumerate(tqdm(dataloader)):
		
		x = x.to(dvc)
		y_true = y_true.to(dvc)
		x_pths += list(x_pth)

		n_samples = x.shape[0]
		
		use_cutmix = False
		if use_cutmix:
			beta = 1.0
			lam = np.random.beta(beta, beta)
			rand_index = torch.randperm(x.shape[0]).cuda()
			y_true_a = y_true
			y_true_b = y_true[rand_index]
			bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape, lam)
			x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
			# adjust lambda to exactly match pixel ratio
			lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[-1] * x.shape[-2]))
			
		output = model(x)
		
		y_preds = torch.nn.Sigmoid()(output.logits.float())
		y_preds = y_preds >= 0.5

		if use_cutmix:
			losses_batch = loss_fn(output.logits, y_true_a) * lam + loss_fn(output.logits, y_true_b) * (1. - lam)
		else:
			losses_batch = loss_fn(output.logits, y_true)
		
		loss_batch = losses_batch / gradient_acc_steps
		
		y_true_all[idx:(idx+n_samples), :] = y_true.detach().cpu().numpy()
		y_pred_all[idx:(idx+n_samples), :] = y_preds.detach().cpu().numpy()
		loss_all[idx_batch] = loss_batch.detach().cpu().numpy()

		idx += n_samples

		if dataset_split == "train":
			
			loss_batch.backward()
			
			if ((idx_batch + 1) % gradient_acc_steps == 0) or (idx_batch + 1 == len(dataloader)):
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
				if scheduler is not None:
					scheduler.step()
				optimizer.zero_grad()

		# break

	emr, hd, prec, rec, f1_score = compute_metrics(y_true_all, y_pred_all, loss_all, dataset_split, epoch, labels_classes, do_log_wandb)
	
	return emr, hd, prec, rec, f1_score, y_true_all, y_pred_all, x_pths


def prepare_dataset(dataset_name, do_only_first_batch=False, do_download=True):

	ds_root = f"data/{dataset_name}"

	img_root = ds_root + "/train"

	df = pd.read_csv(f"{ds_root}/train/_classes.csv")
	all_data = np.asarray(df)

	if do_only_first_batch:
		all_data = all_data[:20, :]

	n_classes = all_data[:, 1:].shape[1]

	facade_labels = list(df.columns)[1:]

	print("unique labels: ", facade_labels)
	print("total number of samples: ", all_data.shape[0])
	print("# of different classes: ", n_classes)

	class_cnts = np.count_nonzero(all_data[:, 1:], axis=0)
	class_weights = class_cnts.sum() / (n_classes * class_cnts)

	print("class distribution:", class_cnts)
	print("class_weights: ", class_weights)

	print("class distribution:", class_cnts)
	print("class_weights: ", class_weights)

	return all_data, class_weights, n_classes, img_root, facade_labels


def prepare_dataloader(all_data, batch_size, randaugm_m, rand_seed, img_root, image_processor):

	all_train, all_test = train_test_split(all_data, test_size=0.1)

	print("train class distribution:", np.count_nonzero(all_train[:, 1:], axis=0))
	print("test class distribution:", np.count_nonzero(all_test[:, 1:], axis=0))

	train_dataset = BuildingFacadeDataset(all_train, 640, 400, img_root, "train", image_processor, randaugm_n=2, randaugm_m=randaugm_m, do_augm=True)
	test_dataset = BuildingFacadeDataset(all_test, 6400, 400, img_root, "test", image_processor=image_processor)

	print("size train: ", len(train_dataset))
	print("size test: ", len(test_dataset))

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, generator=get_tgseed(rand_seed), shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, generator=get_tgseed(rand_seed), shuffle=True)

	return train_dataloader, train_dataset, test_dataloader, test_dataset


def load_model(model_name, n_classes, dvc):
	
	image_processor = AutoImageProcessor.from_pretrained(model_name)
	
	model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=n_classes, problem_type="multi_label_classification", ignore_mismatched_sizes=True)
	model.to(dvc)

	return model, image_processor


def run_training_loop(model, train_dataloader, train_dataset, test_dataloader, test_dataset, n_epochs, n_classes, gradient_acc_steps, optimizer, scheduler, loss_fn, facade_labels, dvc, do_log_wandb):

	for epoch in range(n_epochs):

		print(f"\n\n>>> EPOCH {epoch} ...\n")
		
		model.train()
		do_epoch(model, train_dataloader, train_dataset, "train", n_classes, gradient_acc_steps, optimizer, scheduler, loss_fn, epoch, facade_labels, dvc, do_log_wandb)
		
		with torch.no_grad():
			model.eval()
			emr, hd, prec, rec, f1_score, y_true, y_pred, x_pths = do_epoch(model, test_dataloader, test_dataset, "test", n_classes, gradient_acc_steps, optimizer, scheduler, loss_fn, epoch, facade_labels, dvc, do_log_wandb)

	return emr, hd, prec, rec, f1_score, y_true, y_pred, x_pths


def save_run_params(filename, learning_rate, gradient_acc_steps, weight_decay, randaugm_m, score):

	with open(filename, "a") as fp:
		data = [ learning_rate, gradient_acc_steps, weight_decay, randaugm_m, score ]
		fp.write(','.join([ str(x) for x in data ]) + "\n")


def get_result_rows(x_pths, y_true, y_pred):
	data = []
	for i in range(y_true.shape[0]):
		row = [ x_pths[i] ]
		for j in range(y_true.shape[1]):
			row.append(y_true[i, j])
		for k in range(y_pred.shape[1]):
			row.append(y_pred[i, k])
		data.append(row)
	return data


def load_model_and_get_cams(pretrained_model_pth, model, test_dataset):
	
	model.load_state_dict(torch.load(pretrained_model_pth))
	model.eval()

	model = HuggingfaceToTensorModelWrapper(model)

	#cam_extractor = SmoothGradCAMpp(model)

	x, x_pth, y_true = test_dataset.__getitem__(0)
	x = x.unsqueeze(0)
	x = x.cuda()

	output = model(x)

	print(output.shape)

	cam_extractor = SmoothGradCAMpp(model)

	print(cam_extractor)

	with SmoothGradCAMpp(model) as cam_extractor:
		output = model(x)

		y_preds = torch.nn.Sigmoid()(output.float())
		y_preds = y_preds >= 0.5
		y_preds = y_preds[0, :] # single batch

		activation_map = cam_extractor(torch.nonzero(y_true, as_tuple=True)[0][0].item(), y_preds)
		#activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
		plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off')
		plt.tight_layout()
		plt.show()


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


def train_model(model_name, dataset_name, n_epochs, run_id, learning_rate, batch_size, gradient_acc_steps, weight_decay, randaugm_m, rand_seed, do_log_wandb, do_only_first_batch=False, filename=None, save_model=False, save_results=False, pretrained_model_pth=None):

	print("LR:", learning_rate)
	print("BS:", batch_size)
	print("GAC:", gradient_acc_steps)

	wandb_exp = wandb.init(entity="cea", project="cea-facades", name=run_id, mode="disabled" if not do_log_wandb else "online")
	dvc = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

	set_seeds(rand_seed)

	all_data, class_weights, n_classes, img_root, facade_labels = prepare_dataset(dataset_name, do_only_first_batch, do_download=False)

	model, image_processor = load_model(model_name, n_classes, dvc)

	train_dataloader, train_dataset, test_dataloader, test_dataset = prepare_dataloader(all_data, batch_size, randaugm_m, rand_seed, img_root, image_processor)

	if pretrained_model_pth is None:
		if weight_decay is None:
			optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
		else:
			optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

		total_n_steps = (len(train_dataloader) / gradient_acc_steps) * n_epochs
		scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_n_steps*0.1, num_training_steps=total_n_steps)

		loss_fn = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(class_weights, device=dvc))

		emr, hd, prec, rec, f1_score, y_true, y_pred, x_pths = run_training_loop(model, train_dataloader, train_dataset, test_dataloader, test_dataset, n_epochs, n_classes, gradient_acc_steps, optimizer, scheduler, loss_fn, facade_labels, dvc, do_log_wandb)
	else:
		load_model_and_get_cams(pretrained_model_pth, model, test_dataset)

	if filename is not None:
		save_run_params(filename, learning_rate, gradient_acc_steps, weight_decay, randaugm_m, f1_score)

	if save_model:
		torch.save(model.state_dict(), "./trained-model.pt")

	if save_results:
		df = pd.DataFrame(get_result_rows(x_pths, y_true, y_pred), columns=[ "filename" ] + facade_labels + [ "pred_" + i for i in facade_labels ])
		df.to_csv(f'./results_testset_{dataset_name}.csv', index=False)

	return emr, hd, prec, rec, f1_score, y_true, y_pred
