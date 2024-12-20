import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import pandas as pd
import argparse
import codecs
import random
import numpy as np
from scipy.stats import pearsonr, spearmanr

import torch,sys
from custom_eval.empathy_intent_personality_classifier import CustomClassifier


def custom_evalutions(generation,result_path):
	parser = argparse.ArgumentParser("Inference")
	
	parser.add_argument("--seed_val", default=12, type=int, help="Seed value")
	parser.add_argument("--Emotion_model_path", type=str, default='src/custom_eval/pretrained_signals/emotion_roberta-base_epochs15_lr2e-05_dropout0.1_warmup0_seed12/model.pt', help="path to emotion classification model")
	parser.add_argument("--ER_model_path", type=str,default='src/custom_eval/pretrained_signals/emotion_react_roberta-base_epochs4_lr2e-05_dropout0.1_warmup0_seed12/model.pt', help="path to emotion react model")
	parser.add_argument("--IP_model_path", type=str, default='src/custom_eval/pretrained_signals/interpretations_studio-ousia_luke-base_epochs4_lr2e-05_dropout0.1_warmup0_seed12/model.pt',help="path to interpretation model")
	parser.add_argument("--EX_model_path", type=str, default='src/custom_eval/pretrained_signals/explorations_roberta-base_epochs5_lr2e-05_dropout0.1_warmup0_seed12/model.pt', help="path to exploration model")
	parser.add_argument("--EI_model_path", type=str, default='src/custom_eval/pretrained_signals/intent_studio-ousia_luke-base_epochs7_lr2e-05_dropout0.1_warmup0_seed12/model1.pt', help="path to empathetic intent model")
	parser.add_argument("--Big5_model_path", type=str, default='src/custom_eval/pretrained_signals/big5_preprocessing_hidden16_mse_lr2e-05_pretrainstudio-ousia_luke-base_dropout0.2_warmup100/model.pt', help="path to personality model")
	parser.add_argument("--MBTI_E_model_path", type=str, default='src/custom_eval/pretrained_signals/MBTI_introverted_regression_mse_hidden16_lr1e-05_batch_size120_pretrainstudio-ousia_luke-base_dropout0.1_warmup100/model.pt', help="path to MBTI Extroversion model")
	parser.add_argument("--MBTI_thinking_model_path", type=str, default='src/custom_eval/pretrained_signals/MBTI_thinking_regression_mse_hidden16_lr1e-05_batch_size128_pretrainstudio-ousia_luke-base_dropout0.1_warmup100/model.pt', help="path to MBTI Thinking model")
	args = parser.parse_args()

	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		print('No GPU available, using the CPU instead.')
		device = torch.device("cpu")

	random.seed(args.seed_val)
	np.random.seed(args.seed_val)
	torch.manual_seed(args.seed_val)
	torch.cuda.manual_seed_all(args.seed_val)
	
	input_file=result_path.split('/')[-1]
	output_path= os.path.join('src/custom_eval/output',input_file)
	if not os.path.exists(output_path):
		os.makedirs(output_path)
  
	generated=generation["generated"].astype(str).tolist()
	true=generation["label"].astype(str).tolist()
		
	generated=[item.replace('spk2: ','').replace('[SPK2] ','') for item in generated]
	true=[item.replace('spk2: ','').replace('[SPK2] ','') for item in true]

	print(generated[0])
	print(true[0])
	assert len(generated)==len(true)

	customclassifier = CustomClassifier(device,
							Emotion_model_path = args.Emotion_model_path,
							ER_model_path = args.ER_model_path, 
							IP_model_path = args.IP_model_path,
							EX_model_path = args.EX_model_path,
							EI_model_path = args.EI_model_path,
							Personality_model_path = args.Big5_model_path,
							MBTI_E_model_path = args.MBTI_E_model_path,
							MBTI_thinking_model_path = args.MBTI_thinking_model_path
							)

	output_file=open(os.path.join(output_path,'generated_annotated_p0.8_t0.7.csv'),'w')
	csv_writer = csv.writer(output_file, delimiter=',', quotechar='"')

	predictions_ER_g, predictions_IP_g, predictions_EX_g, predictions_EI_g, predictions_big5_g, predictions_mbti_e_g, predictions_mbti_t_g,\
				predictions_ER_t, predictions_IP_t, predictions_EX_t, predictions_EI_t, predictions_big5_t, predictions_mbti_e_t, predictions_mbti_t_t, \
				accuracies_Emotion, accuracies_ER, accuracies_IP, accuracies_EX, accuracies_EI, pearson_personality, pearson_mbti_e, pearson_mbti_t= customclassifier.predict_empathy_intent_personality([generated, true])
	
	csv_writer.writerow(['Emo_acc','ER_acc','IP_acc','EX_acc','EI_acc','big5_pearson','big5_p','mbti_intro_pearson','intro_p','mbti_t_pearson','t_p'])
	csv_writer.writerow([format(accuracies_Emotion*100, '.2f'), format(accuracies_ER*100,'.2f'), format(accuracies_IP*100,'.2f'), format(accuracies_EX*100,'.2f'), format(accuracies_EI*100,'.2f'),\
						format(pearson_personality[0],'.4f'),pearson_personality[1], format(pearson_mbti_e[0],'.4f'),pearson_mbti_e[1], format(pearson_mbti_t[0],'.4f'),pearson_mbti_t[1]
						])

	csv_writer.writerow(['id','generated','true','ER_g','ER_t','IP_g','IP_t','EX_g','EX_t','EI_g','EI_t','big5_g','big5_t','mbti_i_g','mbti_i_t','mbti_t_g','mbti_t_t'])
	for i in range(len(generated)):
		csv_writer.writerow([i, generated[i], true[i], predictions_ER_g[i], predictions_ER_t[i], predictions_IP_g[i],predictions_IP_t[i], predictions_EX_g[i], predictions_EX_t[i], predictions_EI_g[i],predictions_EI_t[i],\
							predictions_big5_g[i],predictions_big5_t[i], predictions_mbti_e_g[i], predictions_mbti_e_t[i], predictions_mbti_t_g[i], predictions_mbti_t_t[i]])
	output_file.close()