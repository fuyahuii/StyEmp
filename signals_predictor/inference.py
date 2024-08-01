import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import pandas as pd
import argparse
import codecs
import random
import numpy as np
from scipy.stats import pearsonr, spearmanr

import torch
from empathy_intent_personality_classifier import CustomClassifier


parser = argparse.ArgumentParser("Inference")
parser.add_argument("--input_path", type=str, default='responses/style_empathy_batch_size_64_lr_5e-05_warmup_0_speaker_30_empathy_30_dataset4_addcontext_False_con',help="path to input test data")
parser.add_argument("--output_path", type=str, default='output/style_empathy_batch_size_64_lr_5e-05_warmup_0_speaker_30_empathy_30_dataset4_addcontext_False_con',help="output file path")
parser.add_argument("--seed_val", default=12, type=int, help="Seed value")
# parser.add_argument("--base_model", default="roberta-base", help="base model, choose from [studio-ousia/luke-base, roberta-base]")
parser.add_argument("--Emotion_model_path", type=str, default='Pretrained/emotion_roberta-base_epochs15_lr2e-05_dropout0.1_warmup0_seed12/model.pt', help="path to emotion classification model")
parser.add_argument("--ER_model_path", type=str,default='Pretrained/emotion_react_roberta-base_epochs4_lr2e-05_dropout0.1_warmup0_seed12/model.pt', help="path to emotion react model")
parser.add_argument("--IP_model_path", type=str, default='Pretrained/interpretations_studio-ousia_luke-base_epochs4_lr2e-05_dropout0.1_warmup0_seed12/model.pt',help="path to interpretation model")
parser.add_argument("--EX_model_path", type=str, default='Pretrained/explorations_roberta-base_epochs5_lr2e-05_dropout0.1_warmup0_seed12/model.pt', help="path to exploration model")
parser.add_argument("--EI_model_path", type=str, default='Pretrained/intent_studio-ousia_luke-base_epochs7_lr2e-05_dropout0.1_warmup0_seed12/model1.pt', help="path to empathetic intent model")
parser.add_argument("--Big5_model_path", type=str, default='Pretrained/big5_preprocessing_hidden16_mse_lr2e-05_pretrainstudio-ousia_luke-base_dropout0.2_warmup100/model.pt', help="path to personality model")
parser.add_argument("--MBTI_E_model_path", type=str, default='Pretrained/MBTI_introverted_regression_mse_hidden16_lr1e-05_batch_size120_pretrainstudio-ousia_luke-base_dropout0.1_warmup100/model.pt', help="path to MBTI Extroversion model")
parser.add_argument("--MBTI_thinking_model_path", type=str, default='Pretrained/MBTI_thinking_regression_mse_hidden16_lr1e-05_batch_size128_pretrainstudio-ousia_luke-base_dropout0.1_warmup100/model.pt', help="path to MBTI Thinking model")
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
    
input_df = pd.read_csv(os.path.join(args.input_path,'generated_p0.8_t0.7.csv'))
generated=input_df["generated"].astype(str).tolist()
true=input_df["label"].astype(str).tolist()

# generated=input_df["pred_resp"].astype(str).tolist()
# true=input_df["gold_resp"].astype(str).tolist()

# with open(os.path.join(args.input_path,'hyp_dvg'), 'r') as f:
#     generation=f.readlines()
# generated=[item.strip() for item in generation]

# with open(os.path.join(args.input_path,'ref_cem.txt'), 'r') as f:
#     true=f.readlines()
# true=[item.strip() for item in true]
    
generated=[item.replace('spk2: ','').replace('[SPK2] ','') for item in generated]
true=[item.replace('spk2: ','').replace('[SPK2] ','') for item in true]

print(generated[0])
print(true[0])
assert len(generated)==len(true)

personality_scores=pd.read_csv(os.path.join('responses/test_dataset4.csv'))
big5_e=np.array(personality_scores["big5_e"].astype(float).tolist())
mbti_intro=np.array(personality_scores["mbti_intro"].astype(float).tolist())
mbti_thinking=np.array(personality_scores["mbti_thinking"].astype(float).tolist())

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

output_file=open(os.path.join(args.output_path,'generated_annotated_p0.8_t0.7.csv'),'w')
csv_writer = csv.writer(output_file, delimiter=',', quotechar='"')

predictions_ER_g, predictions_IP_g, predictions_EX_g, predictions_EI_g, predictions_big5_g, predictions_mbti_e_g, predictions_mbti_t_g,\
      		predictions_ER_t, predictions_IP_t, predictions_EX_t, predictions_EI_t, predictions_big5_t, predictions_mbti_e_t, predictions_mbti_t_t, \
            accuracies_Emotion, accuracies_ER, accuracies_IP, accuracies_EX, accuracies_EI, pearson_personality, pearson_mbti_e, pearson_mbti_t= customclassifier.predict_empathy_intent_personality([generated, true])

pearson_big5_e=pearsonr(predictions_big5_g, big5_e)
pearson_mbti_intro=pearsonr(predictions_mbti_e_g, mbti_intro)
pearson_mbti_thinking=pearsonr(predictions_mbti_t_g, mbti_thinking)
print("pearson_big5_e: ", pearson_big5_e)
print("pearson_mbti_e: ", pearson_mbti_intro)
print("pearson_mbti_t: ", pearson_mbti_thinking)

# pearson_big5_e are the pearson correlation coefficient between generated and retrieved big5_e scores
  
csv_writer.writerow(['Emo_acc','ER_acc','IP_acc','EX_acc','EI_acc','big5_pearson','big5_p','mbti_intro_pearson','intro_p','mbti_t_pearson','t_p'])
csv_writer.writerow([format(accuracies_Emotion*100, '.2f'), format(accuracies_ER*100,'.2f'), format(accuracies_IP*100,'.2f'), format(accuracies_EX*100,'.2f'), format(accuracies_EI*100,'.2f'),\
    				format(pearson_personality[0],'.4f'),pearson_personality[1], format(pearson_mbti_e[0],'.4f'),pearson_mbti_e[1], format(pearson_mbti_t[0],'.4f'),pearson_mbti_t[1],\
            		format(pearson_big5_e[0],'.4f'),pearson_big5_e[1], format(pearson_mbti_intro[0],'.4f'),pearson_mbti_intro[1],format(pearson_mbti_thinking[0],'.4f'),pearson_mbti_thinking[1]])

csv_writer.writerow(['id','generated','true','ER_g','ER_t','IP_g','IP_t','EX_g','EX_t','EI_g','EI_t','big5_g','big5_t','mbti_i_g','mbti_i_t','mbti_t_g','mbti_t_t'])
for i in range(len(generated)):
	csv_writer.writerow([i, generated[i], true[i], predictions_ER_g[i], predictions_ER_t[i], predictions_IP_g[i],predictions_IP_t[i], predictions_EX_g[i], predictions_EX_t[i], predictions_EI_g[i],predictions_EI_t[i],\
     					predictions_big5_g[i],predictions_big5_t[i], predictions_mbti_e_g[i], predictions_mbti_e_t[i], predictions_mbti_t_g[i], predictions_mbti_t_t[i]])
output_file.close()