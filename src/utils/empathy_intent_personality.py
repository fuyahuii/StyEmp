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

from models.annotations import Empathy_Intent_Encoder, PersonalityEncoder


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
        return {
            'input_ids_generated':encoding_generated['input_ids'].flatten(),
            'attention_mask_generated':encoding_generated['attention_mask'].flatten(),
            'generated':input['generated'],
        }
       
class EmotionClassifier():

	def __init__(self, 
			device,
			Emotion_model_path = './src/custom_eval/pretrained_signals/emotion_roberta-base_epochs15_lr2e-05_dropout0.1_warmup0_seed12/model.pt',
			batch_size=1024):
		
		self.batch_size = batch_size
		self.device = device
		self.tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-base')
		self.base_model_roberta = AutoModel.from_pretrained("roberta-base")

		self.model_Emotion=Empathy_Intent_Encoder(base_model="roberta-base",hidden_dropout_prob=0.1,num_labels=32)
		self.model_Emotion.load_state_dict(torch.load(Emotion_model_path))
		
		self.model_Emotion.to(self.device)
  
	def predict_emotion(self, generated):
		inference_data=[]
		for i in range(len(generated)):
			inference_data.append({'generated':generated[i]})
		inference_dataset_roberta=Dataset(inference_data,self.tokenizer_roberta,max_len=512)
		dataloader_roberta=DataLoader(inference_dataset_roberta,batch_size=self.batch_size,shuffle=False)
		
		self.model_Emotion.eval()

		emotion_embeddings=[]
		for batch in dataloader_roberta:
			input_ids_generated= batch["input_ids_generated"].to(self.device)
			input_mask_generated = batch["attention_mask_generated"].to(self.device)

			with torch.no_grad():
				emotion_embedding,_=self.model_Emotion(input_ids=input_ids_generated, attention_mask=input_mask_generated)
			emotion_embeddings.extend(emotion_embedding.detach().cpu().numpy().tolist())
   
		return emotion_embeddings

class PersonalityClassifier():

	def __init__(self, 
			device,
			Personality_model_path = './src/custom_eval/pretrained_signals/big5_preprocessing_hidden16_mse_lr2e-05_pretrainstudio-ousia_luke-base_dropout0.2_warmup100/model.pt',
			MBTI_intro_model_path='./src/custom_eval/pretrained_signals/MBTI_introverted_regression_mse_hidden16_lr1e-05_batch_size120_pretrainstudio-ousia_luke-base_dropout0.1_warmup100/model.pt',
			MBTI_thinking_model_path='./src/custom_eval/pretrained_signals/MBTI_thinking_regression_mse_hidden16_lr1e-05_batch_size128_pretrainstudio-ousia_luke-base_dropout0.1_warmup100/model.pt',
			batch_size=1024):
		
		self.batch_size = batch_size
		self.device = device
		self.tokenizer_luke = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
		self.base_model_luke=AutoModel.from_pretrained("studio-ousia/luke-base")

		self.model_Personality=PersonalityEncoder(base_model="studio-ousia/luke-base",hidden_dropout_prob=0.2,num_labels=1,hidden_size=16,num_task=5)
		self.model_Personality=nn.DataParallel(self.model_Personality)
		self.model_Personality.load_state_dict(torch.load(Personality_model_path))

		self.model_MBTI_Intro=PersonalityEncoder(base_model="studio-ousia/luke-base",hidden_dropout_prob=0.1,num_labels=1,hidden_size=16,num_task=1)
		self.model_MBTI_Intro=nn.DataParallel(self.model_MBTI_Intro)
		self.model_MBTI_Intro.load_state_dict(torch.load(MBTI_intro_model_path))
  
		self.model_MBTI_thinking=PersonalityEncoder(base_model="studio-ousia/luke-base",hidden_dropout_prob=0.1,num_labels=1,hidden_size=16,num_task=1)
		self.model_MBTI_thinking=nn.DataParallel(self.model_MBTI_thinking)
		self.model_MBTI_thinking.load_state_dict(torch.load(MBTI_thinking_model_path))
		
		self.model_Personality.to(self.device)
		self.model_MBTI_Intro.to(self.device)
		self.model_MBTI_thinking.to(self.device)
  
	def predict_personality(self, generated):
		inference_data=[]
		for i in range(len(generated)):
			inference_data.append({'generated':generated[i]})
		
		inference_dataset_luke=Dataset(inference_data,self.tokenizer_luke,max_len=512)
		dataloader_luke=DataLoader(inference_dataset_luke,batch_size=self.batch_size,shuffle=False)

		self.model_Personality.eval()
		self.model_MBTI_Intro.eval()
		self.model_MBTI_thinking.eval()

		predictions_big5_g_e, predictions_mbti_g_intro, perdictions_mbti_g_thinking=[],[],[]
		
		for batch in dataloader_luke:
			input_ids_generated= batch["input_ids_generated"].to(self.device)
			input_mask_generated = batch["attention_mask_generated"].to(self.device)
			
			with torch.no_grad():
				logits_personality_g=self.model_Personality(input_ids=input_ids_generated, attention_mask=input_mask_generated)
				logits_mbti_intro_g=self.model_MBTI_Intro(input_ids=input_ids_generated, attention_mask=input_mask_generated)
				logits_mbti_thinking_g=self.model_MBTI_thinking(input_ids=input_ids_generated, attention_mask=input_mask_generated)

			logits_personality_g=[logits.detach().cpu().numpy().tolist() for logits in logits_personality_g]
			predictions_big5_g_e.extend(logits_personality_g[2]) # 2 is the index of the big5 personality trait (extraversion)

			logits_mbti_intro_g=[logits.detach().cpu().numpy().tolist() for logits in logits_mbti_intro_g]
			predictions_mbti_g_intro.extend(logits_mbti_intro_g[0]) 

			logits_mbti_thinking_g=[logits.detach().cpu().numpy().tolist() for logits in logits_mbti_thinking_g]
			perdictions_mbti_g_thinking.extend(logits_mbti_thinking_g[0])
   
		return predictions_big5_g_e, predictions_mbti_g_intro, perdictions_mbti_g_thinking

class EmpathyClassifier():

	def __init__(self, 
			device,
			ER_model_path = './src/custom_eval/pretrained_signals/emotion_react_roberta-base_epochs4_lr2e-05_dropout0.1_warmup0_seed12/model.pt', 
			IP_model_path = './src/custom_eval/pretrained_signals/interpretations_studio-ousia_luke-base_epochs4_lr2e-05_dropout0.1_warmup0_seed12/model.pt',
			EX_model_path = './src/custom_eval/pretrained_signals/explorations_roberta-base_epochs5_lr2e-05_dropout0.1_warmup0_seed12/model.pt',
			EI_model_path = './src/custom_eval/pretrained_signals/intent_studio-ousia_luke-base_epochs7_lr2e-05_dropout0.1_warmup0_seed12/model1.pt',
			batch_size=1024):
		
		self.batch_size = batch_size
		self.device = device
		self.tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-base')
		self.tokenizer_luke = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
		self.base_model_roberta = AutoModel.from_pretrained("roberta-base")
		self.base_model_luke=AutoModel.from_pretrained("studio-ousia/luke-base")

		self.model_ER = Empathy_Intent_Encoder(base_model="roberta-base",hidden_dropout_prob=0.1,num_labels=2)
		self.model_IP = Empathy_Intent_Encoder(base_model="studio-ousia/luke-base",hidden_dropout_prob=0.1,num_labels=2)
		self.model_EX = Empathy_Intent_Encoder(base_model="roberta-base",hidden_dropout_prob=0.1,num_labels=2)
		self.model_EI = Empathy_Intent_Encoder(base_model="studio-ousia/luke-base",hidden_dropout_prob=0.1,num_labels=9)
		
		self.model_ER.load_state_dict(torch.load(ER_model_path))
		self.model_IP.load_state_dict(torch.load(IP_model_path))
		self.model_EX.load_state_dict(torch.load(EX_model_path))
		self.model_EI.load_state_dict(torch.load(EI_model_path))
		
		self.model_ER.to(self.device)
		self.model_IP.to(self.device)
		self.model_EX.to(self.device)
		self.model_EI.to(self.device)
	
  
	def predict_empathy_intent(self, generated):
		inference_data=[]
		for i in range(len(generated)):
			inference_data.append({'generated':generated[i]})
		inference_dataset_roberta=Dataset(inference_data,self.tokenizer_roberta,max_len=512)
		dataloader_roberta=DataLoader(inference_dataset_roberta,batch_size=self.batch_size,shuffle=False)
		inference_dataset_luke=Dataset(inference_data,self.tokenizer_luke,max_len=512)
		dataloader_luke=DataLoader(inference_dataset_luke,batch_size=self.batch_size,shuffle=False)

		self.model_ER.eval()
		self.model_IP.eval()
		self.model_EX.eval()
		self.model_EI.eval()

		predictions_ER_g, predictions_IP_g, predictions_EX_g, predictions_EI_g=[],[],[],[]
		for batch in dataloader_roberta:
			input_ids_generated= batch["input_ids_generated"].to(self.device)
			input_mask_generated = batch["attention_mask_generated"].to(self.device)

			with torch.no_grad():
				_,logits_empathy_ER_g = self.model_ER(input_ids=input_ids_generated, attention_mask=input_mask_generated)
				_,logits_empathy_EX_g = self.model_EX(input_ids=input_ids_generated, attention_mask=input_mask_generated)
			
			logits_empathy_ER_g = [logits.detach().cpu().numpy().tolist() for logits in logits_empathy_ER_g]
			prediction_ER_g = np.argmax(logits_empathy_ER_g, axis=1).flatten()

			logits_empathy_EX_g = [logits.detach().cpu().numpy().tolist() for logits in logits_empathy_EX_g]
			prediction_EX_g = np.argmax(logits_empathy_EX_g, axis=1).flatten()

			predictions_ER_g.extend(prediction_ER_g)
			predictions_EX_g.extend(prediction_EX_g)
		
		for batch in dataloader_luke:
			input_ids_generated= batch["input_ids_generated"].to(self.device)
			input_mask_generated = batch["attention_mask_generated"].to(self.device)
			
			with torch.no_grad():
				_,logits_empathy_IP_g = self.model_IP(input_ids=input_ids_generated, attention_mask=input_mask_generated)
				_,logits_emotion_intent_g=self.model_EI(input_ids=input_ids_generated, attention_mask=input_mask_generated)

			logits_empathy_IP_g = [logits.detach().cpu().numpy().tolist() for logits in logits_empathy_IP_g]
			prediction_IP_g = np.argmax(logits_empathy_IP_g, axis=1).flatten()
   
			logits_EI_g=[logits.detach().cpu().numpy().tolist() for logits in logits_emotion_intent_g]
			prediction_EI_g=np.argmax(logits_EI_g,axis=1).flatten()
   
			predictions_IP_g.extend(prediction_IP_g)
			predictions_EI_g.extend(prediction_EI_g)
   
		return predictions_ER_g, predictions_IP_g, predictions_EX_g, predictions_EI_g

