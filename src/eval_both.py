import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from eval import eval
from custom_eval.inference import custom_evalutions
import pandas as pd


# result_path='calibration/output/result/style_none_batch_size_64_lr_5e-05_warmup_0_dataset4_calibrationtop_can_num_10_true_weight_1_per_weight_9_lm_weight_1'
# result_path='baseline_responses'
result_path='output/result/style_none_batch_size_64_lr_5e-05_warmup_0_dataset4'
print(result_path)
generated_responses=pd.read_csv(result_path+'/generated_p0.8_t0.7.csv')
# # generated_responses=pd.read_csv(result_path+'/chatgpt_comet_k2.csv')
# eval(generated_responses, result_path)
custom_evalutions(generated_responses,result_path)