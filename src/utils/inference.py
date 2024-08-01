import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import pandas as pd
import argparse
import codecs
import random
import numpy as np

import torch
from empathy_intent_personality_classifier import CustomClassifier


parser = argparse.ArgumentParser("Inference")
parser.add_argument("--input_path", type=str, default='responses/batch_size_64_lr_5e-05_warmup_0_baseline',help="path to input test data")
parser.add_argument("--output_path", type=str, default='output/batch_size_64_lr_5e-05_warmup_0_baseline',help="output file path")
parser.add_argument("--seed_val", default=12, type=int, help="Seed value")
# parser.add_argument("--base_model", default="roberta-base", help="base model, choose from [studio-ousia/luke-base, roberta-base]")
parser.add_argument("--Emotion_model_path", type=str, default='Pretrained/emotion_roberta-base_epochs15_lr2e-05_dropout0.1_warmup0_seed12/model.pt', help="path to emotion classification model")
parser.add_argument("--ER_model_path", type=str,default='Pretrained/emotion_react_roberta-base_epochs4_lr2e-05_dropout0.1_warmup0_seed12/model.pt', help="path to emotion react model")
parser.add_argument("--IP_model_path", type=str, default='Pretrained/interpretations_studio-ousia_luke-base_epochs4_lr2e-05_dropout0.1_warmup0_seed12/model.pt',help="path to interpretation model")
parser.add_argument("--EX_model_path", type=str, default='Pretrained/explorations_roberta-base_epochs5_lr2e-05_dropout0.1_warmup0_seed12/model.pt', help="path to exploration model")
parser.add_argument("--EI_model_path", type=str, default='Pretrained/intent_studio-ousia_luke-base_epochs7_lr2e-05_dropout0.1_warmup0_seed12/model1.pt', help="path to empathetic intent model")
parser.add_argument("--Big5_model_path", type=str, default='Pretrained/big5_preprocessing_hidden16_mse_lr2e-05_pretrainstudio-ousia_luke-base_dropout0.2_warmup100/model.pt', help="path to personality model")

args = parser.parse_args()

if not os.path.exists(args.output_path):
	os.makedirs(args.output_path)
 
if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")

random.seed(args.seed_val)
np.random.seed(args.seed_val)
torch.manual_seed(args.seed_val)
torch.cuda.manual_seed_all(args.seed_val)
    
input_df = pd.read_csv(os.path.join(args.input_path,'generated_0.8.csv'))
generated=input_df["generated"].astype(str).tolist()
true=input_df["label"].astype(str).tolist()

customclassifier = CustomClassifier(device,
						Emotion_model_path = args.Emotion_model_path,
						ER_model_path = args.ER_model_path, 
						IP_model_path = args.IP_model_path,
						EX_model_path = args.EX_model_path,
						EI_model_path = args.EI_model_path,
						Personality_model_path = args.Big5_model_path,
						)

output_file=open(os.path.join(args.output_path,'generated_0.8_annotated.csv'),'w')
csv_writer = csv.writer(output_file, delimiter=',', quotechar='"')

predictions_ER_g, predictions_IP_g, predictions_EX_g, predictions_EI_g, predictions_big5_g, predictions_ER_t, predictions_IP_t, predictions_EX_t, predictions_EI_t, predictions_big5_t,accuracies_Emotion, accuracies_ER, accuracies_IP, accuracies_EX, accuracies_EI, pearson_personality = customclassifier.predict_empathy_intent_personality([generated, true])
csv_writer.writerow(['Emo_acc','ER_acc','IP_acc','EX_acc','EI_acc','big5_acc'])
csv_writer.writerow([accuracies_Emotion, accuracies_ER, accuracies_IP, accuracies_EX, accuracies_EI, pearson_personality])

csv_writer.writerow(['id','generated','true','ER_g','ER_t','IP_g','IP_t','EX_g','EX_t','EI_g','EI_t','big5_g','big5_t'])
for i in range(len(generated)):
	csv_writer.writerow([i, generated[i], true[i], predictions_ER_g[i], predictions_ER_t[i], predictions_IP_g[i],predictions_IP_t[i], predictions_EX_g[i], predictions_EX_t[i], predictions_EI_g[i],predictions_EI_t[i],predictions_big5_g[i],predictions_big5_t[i]])
output_file.close()