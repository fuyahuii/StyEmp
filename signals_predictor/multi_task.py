
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import pandas as pd
import random
import argparse
import tqdm
import torch.nn as nn
import torch
from transformers import AdamW,get_linear_schedule_with_warmup, AutoTokenizer, AutoModel	
from models import PersonalityEncoder as Encoder

# from evaluation_utils import *
from sklearn.metrics import accuracy_score,mean_squared_error,accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from scipy.stats import pearsonr, spearmanr

import logging

parser = argparse.ArgumentParser()

parser.add_argument("--data_folder", default="./MBTI/dataset4", help="path to data folder, choose from [./MBTI/dataset1, ./MBTI_nobig5/split2]")
parser.add_argument("--seed_val", default=12, type=int, help="Seed value")
parser.add_argument("--model_output", default="./MBTI_dataset4_models", help="path to save model")
parser.add_argument("--log_path", default="./MBTI_dataset4_logs", help="path to save logs")
parser.add_argument("--result_path", default="./MBTI_dataset4_results", help="path to save results")
parser.add_argument("--task", default="MBTI", help="task name from [big5, MBTI]")
parser.add_argument("--mbti_type",default=False, help="which type of MBTI to predict, choose from ['introverted', 'intuitive', 'thinking', 'perceiving']")
parser.add_argument("--test_only", action="store_true", help="whether only do test")

parser.add_argument("--head_type", default="classification", help="head type, choose from [regression, classification]")
parser.add_argument("--criterion_type", default="crossentropy", help="criterion for regression, choose from [mse, mae, crossentropy]")
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--dropout", default=0.1, type=float, help="dropout")
parser.add_argument("--batch_size", default=128, type=int, help="batch size")
parser.add_argument("--epochs", default=15, type=int, help="number of the max epochs")
parser.add_argument("--pretrain_model", default="studio-ousia/luke-base", help="base model, choose from [studio-ousia/luke-base, roberta-base]")
parser.add_argument("--max_length", default=512, type=int, help="maximum sequence length")
parser.add_argument("--warmup_steps", default=100, type=int, help="warmup steps")
parser.add_argument("--early_stop", default=2, type=int, help="patience for early stopping")
parser.add_argument("--hidden_size", default=64, type=int, help="hidden size for regression/classification head")

args = parser.parse_args()

data_folder = args.data_folder
seed_val = args.seed_val
model_output= args.model_output
log_path = args.log_path
result_path = args.result_path
task=args.task
mbti_type=args.mbti_type	
test_only=args.test_only
head_type=args.head_type

lr = args.lr
dropout = args.dropout
batch_size = args.batch_size
epochs = args.epochs
pretrain_model = args.pretrain_model
max_length = args.max_length
warmup_steps = args.warmup_steps
early_stop = args.early_stop
criterion_type = args.criterion_type
hidden_size = args.hidden_size

variant = f"{task}_{mbti_type}_{head_type}_{criterion_type}_hidden{hidden_size}_lr{lr}_batch_size{batch_size}_pretrain{pretrain_model}_dropout{dropout}_warmup{warmup_steps}"
# variant = f"{task}_{mbti_type}_hidden{hidden_size}_{criterion_type}_lr{lr}_batch_size{batch_size}_pretrain{pretrain_model}_dropout{dropout}_warmup{warmup_steps}"
log_path = os.path.join(log_path, variant)
model_output_path = os.path.join(model_output, variant)
result_path = os.path.join(result_path, variant)
data_folder = os.path.join(data_folder, f"{mbti_type}")

# log_path="MBTI_dataset3_logs/MBTI_thinking_classification_crossentropy_hidden64_lr1e-05_batch_size128_pretrainroberta-base_dropout0.1_warmup100"
# model_output_path="MBTI_dataset3_models/MBTI_thinking_classification_crossentropy_hidden64_lr1e-05_batch_size128_pretrainroberta-base_dropout0.1_warmup100"
# result_path="MBTI_dataset1_results/MBTI_thinking_classification_crossentropy_hidden64_lr1e-05_batch_size128_pretrainroberta-base_dropout0.1_warmup100"


if not os.path.exists(log_path):
	os.makedirs(log_path)
if not os.path.exists(model_output_path):
    os.makedirs(model_output_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)
    
# log_path="MBTI_nobig5_logs/MBTI_nobig5_hidden256_mse_lr5e-06_batch_size128_pretrainroberta-base_dropout0.1_warmup100"  
# model_output_path="MBTI_nobig5_models/MBTI_nobig5_hidden256_mse_lr5e-06_batch_size128_pretrainroberta-base_dropout0.1_warmup100"
logging.basicConfig(filename=os.path.join(log_path, "log.txt"), level=logging.INFO, 
format='%(asctime)s:%(levelname)s:%(message)s')
logging.info("=====================Args====================")
logging.info("data_folder:"+data_folder)
logging.info("seed_val:"+str(seed_val))
logging.info("model_output_path:"+model_output_path)
logging.info("log_path:"+log_path)
logging.info("result_path:"+result_path)
logging.info("task:"+task)
logging.info("mbti_type:"+str(mbti_type))
logging.info("test_only:"+str(test_only))
logging.info("head_type:"+head_type)
logging.info("criterion_type:"+criterion_type)
logging.info("lr:"+str(lr))
logging.info("dropout:"+str(dropout))
logging.info("batch_size:"+str(batch_size))
logging.info("epochs:"+str(epochs))
logging.info("pretrain model:"+pretrain_model)
logging.info("max_length:"+str(max_length))
logging.info("warmup_steps:"+str(warmup_steps))
logging.info("early_stop:"+str(early_stop))
logging.info("hidden_size:"+str(hidden_size))
logging.info("=============================================")


def load_big5_data(data):
	'''
	Load input dataset

	'''
	Input=data["body"].values.tolist()
	Label_agreeableness=data["agreeableness"].values.tolist()
	Label_conscientiousness=data["conscientiousness"].values.tolist()
	Label_extraversion=data["extraversion"].values.tolist()
	Label_neuroticism=data["neuroticism"].values.tolist()
	Label_openness=data['openness'].values.tolist()

	#normalize the labels to [-1,1]
	Label_agreeableness=(Label_agreeableness-np.mean(Label_agreeableness))/np.std(Label_agreeableness)
	Label_conscientiousness=(Label_conscientiousness-np.mean(Label_conscientiousness))/np.std(Label_conscientiousness)
	Label_extraversion=(Label_extraversion-np.mean(Label_extraversion))/np.std(Label_extraversion)
	Label_neuroticism=(Label_neuroticism-np.mean(Label_neuroticism))/np.std(Label_neuroticism)
	Label_openness=(Label_openness-np.mean(Label_openness))/np.std(Label_openness)

	Labels=[Label_agreeableness,Label_conscientiousness,Label_extraversion,Label_neuroticism,Label_openness]
 
	return Input, Labels

def load_MBTI_data(data):
    
    Input=data["body"].values.tolist()
    Label_Introverted=data["introverted"].values.tolist()
    Label_Intuitive=data["intuitive"].values.tolist()
    Label_Thinking=data["thinking"].values.tolist()
    Label_Perceiving=data["perceiving"].values.tolist()
    Labels=[Label_Introverted,Label_Intuitive,Label_Thinking,Label_Perceiving]
    
    return Input, Labels
   
    
class MultiTaskDataset(torch.utils.data.Dataset):
	def __init__(self, data, tokenizer, max_length,tasks):
		self.data = data
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.task=tasks

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		encoding=self.tokenizer(sample['Input'], return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
		# labels = [sample[f'labels_task{i+1}'] for i in range(self.num_task)]
		labels=[sample[task] for task in self.task]

		return {
			'input_ids': torch.tensor(encoding['input_ids']),
			'attention_mask': torch.tensor(encoding['attention_mask']),
			'labels': torch.tensor(labels)
		}

  
def cal_loss(model,batch, criterion,device):
	input_ids = batch['input_ids'].to(device)
	input_ids = input_ids.squeeze(1)
	attention_mask = batch['attention_mask'].to(device)
	batch_labels = batch['labels'].to(device)
	batch_labels = batch_labels.permute(1,0).to(device)
 
	logits_personality = model(input_ids = input_ids, 
								attention_mask=attention_mask, 
								)
	if head_type=="classification":
		batch_labels=batch_labels.long()
	else:
		batch_labels=batch_labels.float()
		logits_personality=[logits.view(-1) for logits in logits_personality]
	loss = sum([criterion(logits, labels) for logits, labels in zip(logits_personality, batch_labels)])
 
	return loss,logits_personality,batch_labels

def train_valid(model, train_dataloader, valid_dataloader, epochs,criterion,patience,device):
	'''
	Training process

	'''
 
	total_steps = len(train_dataloader) * epochs
	optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

	scheduler = get_linear_schedule_with_warmup(optimizer, 
												num_warmup_steps = warmup_steps,
												num_training_steps = total_steps)

	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	#check if save_model_path exit or not

	if not os.path.exists(model_output_path):
		os.makedirs(model_output_path)
	'''
	Do Training
	'''
	best_valid_loss=float('inf')
	patience=0
	for epoch_i in range(epochs):	
		total_train_loss = 0
		model.train()

		print("the {} epoch".format(epoch_i+1))
		logging.info("------- epoch {}-------".format(epoch_i+1))
  
		for batch in tqdm.tqdm(train_dataloader, desc=f"training"):
			optimizer.zero_grad()
			loss,_,_=cal_loss(model,batch, criterion,device)
			total_train_loss += loss.item()

			loss.backward()
			optimizer.step()
			scheduler.step()

		avg_train_loss = total_train_loss / len(train_dataloader)
		print("Average training loss: {0:.4f}".format(avg_train_loss))
		logging.info("Average training loss: {0:.4f}".format(avg_train_loss))

		print("-----Running Validation-----")		
		total_eval_loss = 0
		model.eval()

		for batch in valid_dataloader:
			with torch.no_grad():
				loss,logits,labels=cal_loss(model,batch, criterion,device)
			total_eval_loss += loss.item()
			
		avg_val_loss=total_eval_loss/len(valid_dataloader)
		print(" Validation MSE: {0:.4f}".format(avg_val_loss))
		logging.info(" Validation MSE: {0:.4f}".format(avg_val_loss))
  
		if avg_val_loss < best_valid_loss:
			best_valid_loss = avg_val_loss
			torch.save(model.state_dict(), os.path.join(model_output_path,"model.pt"))
			print("Best checkpoint saved.")
			logging.info("Best checkpoint saved.")
			patience=0
		else:
			patience+=1
			if patience>early_stop:
				break

def evaluate(model, test_dataloader,criterion, device,median,num_task):
	'''
	Evaluate the model on test dataset

	'''
	print("---------------------Running Testing---------------------")
	logging.info("-------------------Running Testing--------------------")
 
	if os.path.exists(os.path.join(model_output_path,"model.pt")):
		model.eval()

		total_test_loss=0
	
		all_predictions = [[] for _ in range(num_task)]
		all_labels = [[] for _ in range(num_task)]

		for batch in test_dataloader:
			with torch.no_grad():
				loss,logits,labels=cal_loss(model,batch, criterion,device)
			total_test_loss += loss.item()

			for i in range(len(logits)):
				logits[i]=logits[i].cpu().tolist()
				if head_type=="regression":
					all_predictions[i].extend(logits[i])
				elif head_type=="classification":
					all_predictions[i].extend(np.argmax(logits[i], axis=1).flatten().tolist())
				all_labels[i].extend([labels.cpu().tolist() for labels in labels[i]])
	
		avg_test_loss=total_test_loss/len(test_dataloader)
		print("Test loss: {0:.4f}".format(avg_test_loss))
		logging.info("Test loss: {0:.4f}".format(avg_test_loss))

		#save the predictions and labels to csv file
		# for i in range(len(all_predictions)):
		# 	pd.DataFrame(all_predictions[i]).to_csv(os.path.join(result_path,"task{}_predictions.csv".format(i+1)),index=False)
		# 	pd.DataFrame(all_labels[i]).to_csv(os.path.join(result_path,"task{}_labels.csv".format(i+1)),index=False)

		if head_type=="classification":
			accuracy_score_test=[accuracy_score(all_labels[i],all_predictions[i]) for i in range(num_task)]
			balanced_accuracy_score_test=[balanced_accuracy_score(all_labels[i],all_predictions[i]) for i in range(num_task)]
			f1_score_test=[f1_score(all_labels[i],all_predictions[i]) for i in range(num_task)]
			print("average f1_score_test:",np.mean(f1_score_test) )
			logging.info("accuracy_score_test:"+str(accuracy_score_test))
			logging.info("average accuracy_score_test:"+str(np.mean(accuracy_score_test)))
			logging.info("balanced_accuracy_score_test:"+str(balanced_accuracy_score_test))
			logging.info("average balanced_accuracy_score_test:"+str(np.mean(balanced_accuracy_score_test)))
			logging.info("f1_score_test:"+str(f1_score_test))
			logging.info("average f1_score_test:"+str(np.mean(f1_score_test)))

			for i in range(len(all_predictions)):
				confix_matrix=confusion_matrix(all_labels[i],all_predictions[i])
				print("confix_matrix task{}:".format(i+1))
				print(confix_matrix)
				logging.info("confix_matrix task{}:".format(i+1))
				logging.info(confix_matrix)

   
		elif head_type=="regression":
			pearson_corr=[pearsonr(labels, preds) for labels, preds in zip(all_labels, all_predictions)]
			spearman_corr = [spearmanr(labels, preds) for labels, preds in zip(all_labels, all_predictions)]
			#for each task print the pearson correlation and spearman correlation
			for i in range(len(pearson_corr)):
				logging.info("-------------------task{}-------------------------------".format(i+1))
				print("Pearson Correlation for task{}: {:.4f}".format(i+1,pearson_corr[i][0]))
				print("Pearson p-value for task{}: {:.4f}".format(i+1,pearson_corr[i][1]))
				print("Spearman Correlation for task{}: {:.4f}".format(i+1,spearman_corr[i][0]))
				print("Spearman p-value for task{}: {:.4f}".format(i+1,spearman_corr[i][1]))
				print("--------------------------------------------------")
				logging.info("Pearson Correlation for task{}: {:.4f}".format(i+1,pearson_corr[i][0]))
				logging.info("Pearson p-value for task{}: {:.4f}".format(i+1,pearson_corr[i][1]))
				logging.info("Spearman Correlation for task{}: {:.4f}".format(i+1,spearman_corr[i][0]))
				logging.info("Spearman p-value for task{}: {:.4f}".format(i+1,spearman_corr[i][1]))
		
			
			#set the threshold to 0, if the prediction is larger than 0, then the prediction is 1, otherwise 0, calculate the accuracy
			print("-------------------threshold=0-------------------------------")
			logging.info("-------------------threshold=0-------------------------------")
			prediction=[[] for _ in range(num_task)]
			true=[[] for _ in range(num_task)]
			for i in range(len(all_predictions)):
				for j in range(len(all_predictions[i])):
					if all_predictions[i][j]>0:
						prediction[i].append(1)
					else:
						prediction[i].append(0)    
					if all_labels[i][j]>0:
						true[i].append(1)
					else:
						true[i].append(0)
		
			for i in range(len(prediction)):
				confix_matrix=confusion_matrix(true[i],prediction[i])
				print("confix_matrix task{}:".format(i+1))
				print(confix_matrix)
				logging.info("confix_matrix task{}:".format(i+1))
				logging.info(confix_matrix)
			
			f1_score_test=[f1_score(true[i],prediction[i]) for i in range(num_task)]
			accuracy_score_test=[accuracy_score(true[i],prediction[i]) for i in range(num_task)]
			balanced_accuracy_score_test=[balanced_accuracy_score(true[i],prediction[i]) for i in range(num_task)]
			print("average f1_score_test:",np.mean(f1_score_test) )
			print("average accuracy_score_test:",np.mean(accuracy_score_test))	
			print("average balanced_accuracy_score_test:",np.mean(balanced_accuracy_score_test) )
			logging.info("f1_score_test:"+str(f1_score_test))
			logging.info("average f1_score_test:"+str(np.mean(f1_score_test)))
			logging.info("accuracy_score_test:"+str(accuracy_score_test))
			logging.info("average accuracy_score_test:"+str(np.mean(accuracy_score_test)))
			logging.info("balanced_accuracy_score_test:"+str(balanced_accuracy_score_test))
			logging.info("average balanced_accuracy_score_test:"+str(np.mean(balanced_accuracy_score_test)))

			
			print("-------------------threshold=median----------------------------")
			logging.info("-------------------threshold=median--------------------------")
			prediction_median=[[] for _ in range(num_task)]	
			true_median=[[] for _ in range(num_task)]
			for i in range(len(all_predictions)):
				for j in range(len(all_predictions[i])):
					if all_predictions[i][j]>median[i]:
						prediction_median[i].append(1)
					else:
						prediction_median[i].append(0)    
					if all_labels[i][j]>median[i]:
						true_median[i].append(1)
					else:
						true_median[i].append(0)
			f1_score_test_median=[f1_score(true_median[i],prediction_median[i]) for i in range(num_task)]
			accuracy_score_test_median=[accuracy_score(true_median[i],prediction_median[i]) for i in range(num_task)]
			balanced_accuracy_score_test_median=[balanced_accuracy_score(true_median[i],prediction_median[i]) for i in range(num_task)]
			print("average f1_score_test_median:",np.mean(f1_score_test_median) )
			print("average accuracy_score_test_median:",np.mean(accuracy_score_test_median))
			print("average balanced_accuracy_score_test_median:",np.mean(balanced_accuracy_score_test_median))
			logging.info("f1_score_test_median:"+str(f1_score_test_median))
			logging.info("average f1_score_test_median:"+str(np.mean(f1_score_test_median)))
			logging.info("accuracy_score_test_median:"+str(accuracy_score_test_median))
			logging.info("average accuracy_score_test_median:"+str(np.mean(accuracy_score_test_median)))
			logging.info("balanced_accuracy_score_test_median:"+str(balanced_accuracy_score_test_median))
			logging.info("average balanced_accuracy_score_test_median:"+str(np.mean(balanced_accuracy_score_test_median)))
	
		
	else:
		print('No model specified.')
		print('Exiting...')
		exit(-1)

    
def main():
	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		print('No GPU available, using the CPU instead.')
		device = torch.device("cpu")
	
	logging.info("device:"+str(device))
	logging.info("Loading data...")
	
	if task=="big5":
		train=pd.read_csv(os.path.join(data_folder,"train_big5_comment_select_concatenate.csv"))
		valid=pd.read_csv(os.path.join(data_folder,"valid_big5_comment_select_concatenate.csv"))
		test=pd.read_csv(os.path.join(data_folder,"test_big5_comment_select_concatenate.csv"))
		
		train_Input, train_Labels=load_big5_data(train)
		valid_Input, valid_Labels=load_big5_data(valid)
		test_Input, test_Labels=load_big5_data(test)

		train_data, valid_data, test_data = []
		for i in range(len(train_Input)):
			train_data.append({'Input':train_Input[i],'labels_task1':train_Labels[0][i],'labels_task2':train_Labels[1][i],'labels_task3':train_Labels[2][i],'labels_task4':train_Labels[3][i],'labels_task5':train_Labels[4][i]})
		for i in range(len(valid_Input)):
			valid_data.append({'Input':valid_Input[i],'labels_task1':valid_Labels[0][i],'labels_task2':valid_Labels[1][i],'labels_task3':valid_Labels[2][i],'labels_task4':valid_Labels[3][i],'labels_task5':valid_Labels[4][i]})
		for i in range(len(test_Input)):
			test_data.append({'Input':test_Input[i],'labels_task1':test_Labels[0][i],'labels_task2':test_Labels[1][i],'labels_task3':test_Labels[2][i],'labels_task4':test_Labels[3][i],'labels_task5':test_Labels[4][i]})
		median=[np.median(train_Labels[0]),np.median(train_Labels[1]),np.median(train_Labels[2]),np.median(train_Labels[3]),np.median(train_Labels[4])]
		tasks=["labels_task1","labels_task2","labels_task3","labels_task4","labels_task5"]
  
	elif task=="MBTI":
		train=pd.read_csv(os.path.join(data_folder,"train.csv"))
		valid=pd.read_csv(os.path.join(data_folder,"valid.csv"))
		test=pd.read_csv(os.path.join(data_folder,"test.csv"))
		logging.info("train size:"+str(len(train)))
		logging.info("valid size:"+str(len(valid)))
		logging.info("test size:"+str(len(test)))
		
		train_Input, train_Labels=load_MBTI_data(train)
		valid_Input, valid_Labels=load_MBTI_data(valid)
		test_Input, test_Labels=load_MBTI_data(test)
  
		train_data, valid_data, test_data = [],[],[]
		for i in range(len(train_Input)):
			train_data.append({'Input':train_Input[i],'introverted':train_Labels[0][i],'intuitive':train_Labels[1][i],'thinking':train_Labels[2][i],'perceiving':train_Labels[3][i]})
		for i in range(len(valid_Input)):
			valid_data.append({'Input':valid_Input[i],'introverted':valid_Labels[0][i],'intuitive':valid_Labels[1][i],'thinking':valid_Labels[2][i],'perceiving':valid_Labels[3][i]})
		for i in range(len(test_Input)):
			test_data.append({'Input':test_Input[i],'introverted':test_Labels[0][i],'intuitive':test_Labels[1][i],'thinking':test_Labels[2][i],'perceiving':test_Labels[3][i]})
		median=[0.5,0.5,0.5,0.5]
	
		tasks=["introverted","intuitive","thinking","perceiving"]
		if mbti_type:
			tasks=[mbti_type]
			df_0_train=[train_data[i] for i in range(len(train_data)) if train_data[i][mbti_type]==0]
			df_1_train=[train_data[i] for i in range(len(train_data)) if train_data[i][mbti_type]==1]
			df_0_valid=[valid_data[i] for i in range(len(valid_data)) if valid_data[i][mbti_type]==0]
			df_1_valid=[valid_data[i] for i in range(len(valid_data)) if valid_data[i][mbti_type]==1]
			df_0_test=[test_data[i] for i in range(len(test_data)) if test_data[i][mbti_type]==0]
			df_1_test=[test_data[i] for i in range(len(test_data)) if test_data[i][mbti_type]==1]
			print("the number of 0 and 1 in train:",len(df_0_train),len(df_1_train))
			print("the number of 0 and 1 in valid:",len(df_0_valid),len(df_1_valid))
			print("the number of 0 and 1 in test:",len(df_0_test),len(df_1_test))
			logging.info("the number of 0 and 1 in train:"+str(len(df_0_train))+","+str(len(df_1_train)))
			logging.info("the number of 0 and 1 in valid:"+str(len(df_0_valid))+","+str(len(df_1_valid)))
			logging.info("the number of 0 and 1 in test:"+str(len(df_0_test))+","+str(len(df_1_test)))

	num_task=len(tasks)
	print("task:",tasks)
	print("num_task:",num_task)
	logging.info("task:"+str(tasks))
	logging.info("num_task:"+str(num_task))
 
	# tokenizer = RobertaTokenizer.from_pretrained(pretrain_model)
	tokenizer=AutoTokenizer.from_pretrained(pretrain_model)
	dataset_train=MultiTaskDataset(train_data, tokenizer,max_length,tasks=tasks)
	train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
	dataset_valid=MultiTaskDataset(valid_data, tokenizer,max_length,tasks=tasks)
	valid_dataloader = torch.utils.data.DataLoader(dataset_valid,batch_size = batch_size,shuffle=False)
	dataset_test=MultiTaskDataset(test_data, tokenizer,max_length,tasks=tasks)
	test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle=False)
 
	#loss function
	if criterion_type=="mse":
		criterion=nn.MSELoss()
	elif criterion_type=="mae":
		criterion=nn.L1Loss()
	elif criterion_type=="crossentropy":
		criterion=nn.CrossEntropyLoss()
	else:
		print("criterion is not defined")
		exit(-1)
  
	logging.info("Loading model...")
	#load the pretrained model
	# base_model = RobertaModel.from_pretrained(pretrain_model)
	base_model=AutoModel.from_pretrained(pretrain_model)
	
	if head_type=="regression":
		num_label=1
	elif head_type=="classification":
		num_label=2
	model =Encoder(base_model=base_model,hidden_dropout_prob=dropout,num_labels=num_label,hidden_size=hidden_size,num_task=num_task)
	model=nn.DataParallel(model)
	model.to(device)

	if test_only:
		model.load_state_dict(torch.load(os.path.join(model_output_path,"model.pt")))
		evaluate(model, test_dataloader,criterion, device,median,num_task)
	else:
		logging.info("Training...")
		train_valid(model, train_dataloader, valid_dataloader, epochs,criterion,early_stop,device)
		logging.info("Testing...")
		model.load_state_dict(torch.load(os.path.join(model_output_path,"model.pt")))
		evaluate(model, test_dataloader,criterion, device,median,num_task)
		
if __name__ == "__main__":
	main()	