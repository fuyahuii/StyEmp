import torch
import codecs
import numpy as np


import pandas as pd
import re
import csv
import numpy as np
import sys
import argparse

import time

from sklearn.metrics import f1_score

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from models import Empathy_Intent_Encoder, PersonalityEncoder


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data=data
        self.tokenizer=tokenizer
        self.max_len=max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input = self.data[index]
        encoding_generated=self.tokenizer(input['generated'],return_tensors='pt',max_length=self.max_len,truncation=True,padding='max_length')
        encoding_true=self.tokenizer(input['true'],return_tensors='pt',max_length=self.max_len,truncation=True,padding='max_length')
        
        return {
            'input_ids_generated':encoding_generated['input_ids'].flatten(),
            'attention_mask_generated':encoding_generated['attention_mask'].flatten(),
            'input_ids_true':encoding_true['input_ids'].flatten(),
            'attention_mask_true':encoding_true['attention_mask'].flatten(),
            'generated':input['generated'],
            'true':input['true']
        }
        
class CustomClassifier():

	def __init__(self, 
			device,
			Emotion_model_path = 'output/sample.pth',
			ER_model_path = 'output/sample.pth', 
			IP_model_path = 'output/sample.pth',
			EX_model_path = 'output/sample.pth',
			EI_model_path = 'output/sample.pth',
			Personality_model_path = 'output/sample.pth',
			MBTI_E_model_path = 'output/sample.pth',
			MBTI_thinking_model_path = 'output/sample.pth',
			batch_size=1024):
		
		self.batch_size = batch_size
		self.device = device
		self.tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-base')
		self.tokenizer_luke = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
		self.base_model_roberta = AutoModel.from_pretrained("roberta-base")
		self.base_model_luke=AutoModel.from_pretrained("studio-ousia/luke-base")

		self.model_Emotion=Empathy_Intent_Encoder(base_model="roberta-base",hidden_dropout_prob=0.1,num_labels=32)
		self.model_ER = Empathy_Intent_Encoder(base_model="roberta-base",hidden_dropout_prob=0.1,num_labels=2)
		self.model_IP = Empathy_Intent_Encoder(base_model="studio-ousia/luke-base",hidden_dropout_prob=0.1,num_labels=2)
		self.model_EX = Empathy_Intent_Encoder(base_model="roberta-base",hidden_dropout_prob=0.1,num_labels=2)
		self.model_EI = Empathy_Intent_Encoder(base_model="studio-ousia/luke-base",hidden_dropout_prob=0.1,num_labels=9)
		self.model_Personality=PersonalityEncoder(base_model="studio-ousia/luke-base",hidden_dropout_prob=0.2,num_labels=1,hidden_size=16,num_task=5)
		self.model_Personality=nn.DataParallel(self.model_Personality)
		self.model_MBTI_E=PersonalityEncoder(base_model="studio-ousia/luke-base",hidden_dropout_prob=0.1,num_labels=1,hidden_size=16,num_task=1)
		self.model_MBTI_E=nn.DataParallel(self.model_MBTI_E)
		self.model_MBTI_thinking=PersonalityEncoder(base_model="studio-ousia/luke-base",hidden_dropout_prob=0.1,num_labels=1,hidden_size=16,num_task=1)
		self.model_MBTI_thinking=nn.DataParallel(self.model_MBTI_thinking)

		self.model_Emotion.load_state_dict(torch.load(Emotion_model_path))
		self.model_ER.load_state_dict(torch.load(ER_model_path))
		self.model_IP.load_state_dict(torch.load(IP_model_path))
		self.model_EX.load_state_dict(torch.load(EX_model_path))
		self.model_EI.load_state_dict(torch.load(EI_model_path))
		self.model_Personality.load_state_dict(torch.load(Personality_model_path))
		self.model_MBTI_E.load_state_dict(torch.load(MBTI_E_model_path))
		self.model_MBTI_thinking.load_state_dict(torch.load(MBTI_thinking_model_path))
		
		self.model_Emotion.to(self.device)
		self.model_ER.to(self.device)
		self.model_IP.to(self.device)
		self.model_EX.to(self.device)
		self.model_EI.to(self.device)
		self.model_Personality.to(self.device)
		self.model_MBTI_E.to(self.device)
		self.model_MBTI_thinking.to(self.device)
  
  
	def predict_empathy_intent_personality(self, inputs):
		
		generations, trues = inputs
		inference_data=[]
		for i in range(len(generations)):
			generated=generations[i]
			true=trues[i]
			inference_data.append({'true':true,'generated':generated})
		inference_dataset_roberta=Dataset(inference_data,self.tokenizer_roberta,max_len=512)
		dataloader_roberta=DataLoader(inference_dataset_roberta,batch_size=self.batch_size,shuffle=False)
		inference_dataset_luke=Dataset(inference_data,self.tokenizer_luke,max_len=512)
		dataloader_luke=DataLoader(inference_dataset_luke,batch_size=self.batch_size,shuffle=False)

		self.model_Emotion.eval()
		self.model_ER.eval()
		self.model_IP.eval()
		self.model_EX.eval()
		self.model_EI.eval()
		self.model_Personality.eval()
		self.model_MBTI_E.eval()
		self.model_MBTI_thinking.eval()

		predictions_E_t, predictions_E_g, predictions_ER_g, predictions_ER_t, predictions_IP_g, predictions_IP_t, predictions_EX_g, predictions_EX_t, predictions_EI_g, predictions_EI_t, predictions_big5_g, predictions_big5_t, predictions_mbti_e_g, predictions_mbti_e_t, predictions_mbti_t_g,predictions_mbti_t_t=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
		for batch in dataloader_roberta:
			input_ids_generated= batch["input_ids_generated"].to(self.device)
			input_mask_generated = batch["attention_mask_generated"].to(self.device)
			input_ids_true= batch["input_ids_true"].to(self.device)
			input_mask_true = batch["attention_mask_true"].to(self.device)

			with torch.no_grad():
				logits_E_g=self.model_Emotion(input_ids=input_ids_generated, attention_mask=input_mask_generated)
				logits_empathy_ER_g = self.model_ER(input_ids=input_ids_generated, attention_mask=input_mask_generated)
				logits_empathy_EX_g = self.model_EX(input_ids=input_ids_generated, attention_mask=input_mask_generated)

				logits_E_t=self.model_Emotion(input_ids=input_ids_true, attention_mask=input_mask_true)
				logits_empathy_ER_ture = self.model_ER(input_ids=input_ids_true, attention_mask=input_mask_true)
				logits_empathy_EX_ture = self.model_EX(input_ids=input_ids_true, attention_mask=input_mask_true)
			
			logits_E_g=logits_E_g.detach().cpu().numpy().tolist()
			prediction_E_g=np.argmax(logits_E_g,axis=1).flatten()
   
			logits_empathy_ER_g = logits_empathy_ER_g.detach().cpu().numpy().tolist()
			prediction_ER_g = np.argmax(logits_empathy_ER_g, axis=1).flatten()

			logits_empathy_EX_g = logits_empathy_EX_g.detach().cpu().numpy().tolist()
			prediction_EX_g = np.argmax(logits_empathy_EX_g, axis=1).flatten()
		
			logits_E_ture=logits_E_t.detach().cpu().numpy().tolist()
			prediction_E_ture=np.argmax(logits_E_ture,axis=1).flatten()
   
			logits_empathy_ER_ture = logits_empathy_ER_ture.detach().cpu().numpy().tolist()
			prediction_ER_ture = np.argmax(logits_empathy_ER_ture, axis=1).flatten()
   
			logits_empathy_EX_ture = logits_empathy_EX_ture.detach().cpu().numpy().tolist()
			prediction_EX_ture = np.argmax(logits_empathy_EX_ture, axis=1).flatten()

			predictions_E_g.extend(prediction_E_g)
			predictions_ER_g.extend(prediction_ER_g)
			predictions_EX_g.extend(prediction_EX_g)
			predictions_E_t.extend(prediction_E_ture)
			predictions_ER_t.extend(prediction_ER_ture)
			predictions_EX_t.extend(prediction_EX_ture)

		for batch in dataloader_luke:
			input_ids_generated= batch["input_ids_generated"].to(self.device)
			input_mask_generated = batch["attention_mask_generated"].to(self.device)
			input_ids_true= batch["input_ids_true"].to(self.device)
			input_mask_true = batch["attention_mask_true"].to(self.device)
			
			with torch.no_grad():
				logits_empathy_IP_g = self.model_IP(input_ids=input_ids_generated, attention_mask=input_mask_generated)
				logits_emotion_intent_g=self.model_EI(input_ids=input_ids_generated, attention_mask=input_mask_generated)
				logits_personality_g=self.model_Personality(input_ids=input_ids_generated, attention_mask=input_mask_generated)
				logits_mbti_e_g=self.model_MBTI_E(input_ids=input_ids_generated, attention_mask=input_mask_generated)
				logits_mbti_t_g=self.model_MBTI_thinking(input_ids=input_ids_generated, attention_mask=input_mask_generated)
			
				logits_empathy_IP_ture = self.model_IP(input_ids=input_ids_true, attention_mask=input_mask_true)
				logits_emotion_intent_ture=self.model_EI(input_ids=input_ids_true, attention_mask=input_mask_true)
				logits_personality_ture=self.model_Personality(input_ids=input_ids_true, attention_mask=input_mask_true)
				logits_mbti_e_t=self.model_MBTI_E(input_ids=input_ids_true, attention_mask=input_mask_true)
				logits_mbti_t_t=self.model_MBTI_thinking(input_ids=input_ids_true, attention_mask=input_mask_true)

			logits_empathy_IP_g = logits_empathy_IP_g.detach().cpu().numpy().tolist()
			prediction_IP_g = np.argmax(logits_empathy_IP_g, axis=1).flatten()
   
			logits_EI_g=logits_emotion_intent_g.detach().cpu().numpy().tolist()
			prediction_EI_g=np.argmax(logits_EI_g,axis=1).flatten()

			logits_personality_g=[logits.detach().cpu().numpy().tolist() for logits in logits_personality_g]
			logits_mbti_e_g=[logits.detach().cpu().numpy().tolist() for logits in logits_mbti_e_g]
			logits_mbti_t_g=[logits.detach().cpu().numpy().tolist() for logits in logits_mbti_t_g]
	
			logits_empathy_IP_ture = logits_empathy_IP_ture.detach().cpu().numpy().tolist()
			prediction_IP_ture = np.argmax(logits_empathy_IP_ture, axis=1).flatten()
	
			logits_EI_ture=logits_emotion_intent_ture.detach().cpu().numpy().tolist()
			prediction_EI_ture=np.argmax(logits_EI_ture,axis=1).flatten()

			logits_personality_ture=[logits.detach().cpu().numpy().tolist() for logits in logits_personality_ture]
			logits_mbti_e_t=[logits.detach().cpu().numpy().tolist() for logits in logits_mbti_e_t]
			logits_mbti_t_t=[logits.detach().cpu().numpy().tolist() for logits in logits_mbti_t_t]

			predictions_IP_g.extend(prediction_IP_g)
			predictions_EI_g.extend(prediction_EI_g)
			predictions_IP_t.extend(prediction_IP_ture)
			predictions_EI_t.extend(prediction_EI_ture)
			predictions_big5_g.extend(logits_personality_g[2])
			predictions_big5_t.extend(logits_personality_ture[2])#2 is the index of the personality trait extraversion
			predictions_mbti_e_g.extend(logits_mbti_e_g[0])
			predictions_mbti_e_t.extend(logits_mbti_e_t[0])
			predictions_mbti_t_g.extend(logits_mbti_t_g[0])
			predictions_mbti_t_t.extend(logits_mbti_t_t[0])

		accuracies_Emotion=f1_score(predictions_E_t, predictions_E_g, average='weighted')
		accuracies_ER = f1_score(predictions_ER_t, predictions_ER_g, average='weighted')
		accuracies_IP = f1_score(predictions_IP_t, predictions_IP_g, average='weighted')
		accuracies_EX = f1_score(predictions_EX_t, predictions_EX_g, average='weighted')
		accuracies_EI = f1_score(predictions_EI_t, predictions_EI_g, average='weighted')
		predictions_big5_g=np.array(predictions_big5_g).reshape(-1)
		predictions_big5_t=np.array(predictions_big5_t).reshape(-1)
		pearson_personality = pearsonr(predictions_big5_t, predictions_big5_g)
		predictions_mbti_e_g=np.array(predictions_mbti_e_g).reshape(-1)
		predictions_mbti_e_t=np.array(predictions_mbti_e_t).reshape(-1)	
		pearson_mbti_e = pearsonr(predictions_mbti_e_t, predictions_mbti_e_g)
		predictions_mbti_t_g=np.array(predictions_mbti_t_g).reshape(-1)
		predictions_mbti_t_g=np.array(predictions_mbti_t_g).reshape(-1)
		predictions_mbti_t_t=np.array(predictions_mbti_t_t).reshape(-1)
		pearson_mbti_t = pearsonr(predictions_mbti_t_t, predictions_mbti_t_g)
		
		print('Emotion acc:', accuracies_Emotion)
		print('ER acc:', accuracies_ER)
		print('IP acc:', accuracies_IP)
		print('EX acc:', accuracies_EX)
		print('EI acc:', accuracies_EI)
		print('Personality acc:', pearson_personality)
		print('MBTI E acc:', pearson_mbti_e)
		print('MBTI T acc:', pearson_mbti_t)
   
		return(predictions_ER_g, predictions_IP_g, predictions_EX_g, predictions_EI_g, predictions_big5_g, predictions_mbti_e_g, predictions_mbti_t_g,\
      		predictions_ER_t, predictions_IP_t, predictions_EX_t, predictions_EI_t, predictions_big5_t, predictions_mbti_e_t, predictions_mbti_t_t, \
            accuracies_Emotion, accuracies_ER, accuracies_IP, accuracies_EX, accuracies_EI, pearson_personality, pearson_mbti_e, pearson_mbti_t)
