import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import shutil
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F

from utils.generator_utils import encode_example, DataCollator, generate_responses
import wandb
from datasets import load_dataset, load_from_disk, Dataset,DatasetDict
from models.dialogpt import Roberta_GPT2 as RGPT2
from models.dialogpt import StyleRoberta_GPT2 as StyleRGPT2
from models.dialogpt import StyleRoberta_GPT2_Personality as StyleRGPT2_P
from models.dialogpt import StyleRoberta_GPT2_Empathy as StyleRGPT2_E
from models.dialogpt import CustomGPT2LMHeadModel
from models.dialogpt import StyleRoberta_GPT2_calibrate as StyleRGPT2_calibrate
from eval import eval
from custom_eval.inference import custom_evalutions
from utils.calibrate_utils import personality_scorer,personality_loss,RankingLoss

from models.retrieval_explanation import conversation_retrieval,speaker_history_retrieval,empathy_explanation
import sys
import pickle

from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
import json

class CustomTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):

        self.lm_loss_weight = kwargs.pop('lm_loss_weight', 1.0)
        self.per_loss_weight = kwargs.pop('per_loss_weight', 1.0)
        self.true_weight = kwargs.pop('true_weight', 1.0)
        self.top_can_num=kwargs.pop("top_can_num",5)
        
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
    
        outputs = model(**inputs)
        original_loss=outputs.loss  # torch.Size([bz])
        original_logits = outputs.logits  # torch.Size([bz, max_len, vocab]) 
        
        if model.training:    
            batch_size = max(inputs["input_ids"].shape[0] // (self.top_can_num + 1), 1)
            assert batch_size==inputs["input_ids"].shape[0]/(self.top_can_num+1)
            # if inputs["input_ids"].shape[0]/(self.top_can_num+1) != batch_size:
            #     inputs["input_ids"] = inputs["input_ids"][:batch_size*(self.top_can_num+1)]
            #     original_logits = original_logits[:batch_size*(self.top_can_num+1)]
            #     original_loss = original_loss[:batch_size*(self.top_can_num+1)]
            input_ids = inputs["input_ids"].view(batch_size, -1, inputs["input_ids"].shape[-1]) # torch.Size([batch_size, candidate_num+1, max_len])
        
            original_loss=original_loss.view(batch_size,-1) # torch.Size([batch_size, candidate_num+1])
            lm_loss=original_loss[:,0].mean()
            
            original_logits=original_logits.view(batch_size,-1,original_logits.shape[-2],original_logits.shape[-1]) # torch.Size([batch_size, candidate_num+1, max_len, vocab])
            output = F.log_softmax(original_logits, dim=3)
        
            scores=torch.gather(output, 3, input_ids.unsqueeze(-1)).squeeze(-1) # torch.Size([bz, candidate_num+1, max_len])
            scores = torch.mean(scores, dim=2) # torch.Size([bz, candidate_num+1])
  
            # calculate constrastive loss
            personality_loss = RankingLoss(scores, true_weight=self.true_weight)
            personality_loss=personality_loss.to(original_loss.device)
            print("lm_loss: ", lm_loss)
            print("personality_loss: ", personality_loss)
        
            combined_loss = self.lm_loss_weight*lm_loss + self.per_loss_weight*personality_loss
            
            if return_outputs:
                return (combined_loss, outputs)
            
            return combined_loss
        
        else:
            lm_loss=original_loss.mean()
            if return_outputs:
                return (lm_loss, outputs)
            
            return lm_loss


def convert_data(dataset_subset,data="train"):
    # Identify and remove groups with NaN responses
    # dataset_subset['index'] = dataset_subset.index // (candidate_num + 1)
    # groups_with_nan = dataset_subset[pd.isnull(dataset_subset[response_column])]['index'].unique()
    # dataset_subset = dataset_subset[~dataset_subset['index'].isin(groups_with_nan)].reset_index(drop=True)
    
    response_column = 'response' if data == "train" else 'retrieved_response'
    nan_num=0
    for i in range (len(dataset_subset)):
        if pd.isnull(dataset_subset[response_column][i]):
            print("N/A response: ", i, dataset_subset[response_column][i])
            dataset_subset[response_column][i]="N/A"
            nan_num+=1
    print("nan_num: ", nan_num)
    # dataset_subset[response_column]=dataset_subset[response_column] if isinstance(dataset_subset[response_column], str) and dataset_subset[response_column].strip() else "N/A"
    # Replace '[SEP]' with '; ' in concatenated_speaker_history and update empathy_explanation
    dataset_subset['cancatenated_speaker_history'] = dataset_subset['cancatenated_speaker_history'].str.replace('[SEP]', '; ')
    dataset_subset['empathy_explanation'] = dataset_subset['empathy_explanation'] + " The response is: " + dataset_subset[response_column].astype(str)
    
    return dataset_subset

def convert_and_map(dataset_subset,encode_func, tokenizer, roberta_tokenizer,data_type="train"):
    print("data shape: ", len(dataset_subset))
    converted_data = convert_data(dataset_subset,data_type)
    converted_dataset = Dataset.from_pandas(pd.DataFrame(converted_data))
    return converted_dataset.map(encode_func, fn_kwargs={"tokenizer": tokenizer, "roberta_tokenizer": roberta_tokenizer}, num_proc=16)

                  
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    parser= argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="path to the dataset", default="calibration/dataset_calibrate1")
    parser.add_argument("--model_input_path", type=str, help="path to save trained model before calibration ", default="output/model")
    parser.add_argument('--model_output_path', type=str, help="path to save trained model after calibration", default="calibration/output/model")
    parser.add_argument('--result_input_path', type=str, help="path to obtain generated responses before calibration", default="calibration/input")
    parser.add_argument('--result_output_path', type=str, help="path to save generated responses after calibration", default="calibration/output/result")
    parser.add_argument('--temp_path', type=str, help="path to save log", default="calibration/output/temp")
    parser.add_argument('--log_path', type=str, help="path to save log", default="calibration/output/log")
    
    parser.add_argument("--stylizeEncoder", default=True, help="whether to use stylizeEncoder")
    parser.add_argument("--style",type=str, default="both", help="choose from [personality, empathy, both, context,none]")
    parser.add_argument('--personality_reinforcement', default=True, help="whether to use personality reinforcement")
    parser.add_argument('--addcontext', default='False', help="whether to add context slots")
    parser.add_argument('--concontext', default='True', help="whether to add context embeddings")
    parser.add_argument('--diffencoder', default='True', help="whether to use different encoder for style and context")
    
    parser.add_argument('--top_can_num', type=int, default=5)
    parser.add_argument('--batch_size',type=int, default=96) #96 4
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument("--true_weight", type=float, default=0, help="weight for true response rank loss")
    parser.add_argument("--per_weight", type=float, default=5, help="weight for personality loss")
    parser.add_argument("--lm_weight", type=float, default=1, help="weight for lm loss")
    parser.add_argument('--tqdm', default=True,help="whether to use tqdm")
    parser.add_argument('--speaker_slots', type=int, default=25, help="number of speaker slots")
    parser.add_argument('--empathy_slots', type=int, default=25, help="number of empathy slots")
    
    args=parser.parse_args()
    print(args)
    
    save_variable="style_both_batch_size_64_lr_5e-05_warmup_0_speaker_25_empathy_25_addcontext_False_concontext_True_diffencoder_True"
    # save_variable="style_empathy_batch_size_64_lr_5e-05_warmup_0_speaker_20_empathy_20_dataset4_addcontext_True_concontext_False_diffencoder_True"
    # save_variable="style_personality_batch_size_64_lr_5e-05_warmup_0_speaker_20_empathy_20_dataset4_addcontext_True_diffencoder_True"
    # save_variable="style_context_batch_size_64_lr_5e-05_warmup_0_speaker_30_empathy_30_dataset4_addcontext"
    model_input_path=os.path.join(args.model_input_path,save_variable)
    result_input_path=os.path.join(args.result_input_path,save_variable,"num_candidate_5")
    result_temp_path=os.path.join(args.temp_path,save_variable,"num_candidate_5")
    data_path=os.path.join(args.data_path,save_variable+"_calibrate_num_candidate_5")
    
    if not os.path.exists(result_temp_path):
        os.makedirs(result_temp_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    save_variable=save_variable+f"top_can_num_{args.top_can_num}_true_weight_{args.true_weight}_per_weight_{args.per_weight}_lm_weight_{args.lm_weight}_bz_{args.batch_size}"
    model_output_path=os.path.join(args.model_output_path,save_variable)
    
    result_output_path=os.path.join(args.result_output_path,save_variable)
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    if not os.path.exists(result_output_path):
        os.makedirs(result_output_path)
        
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

    if args.stylizeEncoder:
        dataset_path = os.path.join(data_path, "empathetic_dataset_retrieval_calibrate")
 
        if local_rank <= 0 and not os.path.exists(dataset_path):
            print(f"tokenizing data in {dataset_path} ...")
            
            candidates=pd.read_csv(os.path.join(result_input_path,"calibration_p0.9_t0.7_train.csv"))
            generated_responses=candidates["generated"].astype(str).tolist()
            # ground_truth=candidates["label"].astype(str).tolist()
            # candidate_generation=[ground_truth[i]+"\n"+generated_responses[i] for i in range(len(generated_responses))]
            
            candidate_generation = [line.strip() for response in generated_responses for line in response.split("\n")]
            print(f"number of candidates: {len(candidate_generation)}")

            if os.path.exists(os.path.join(result_temp_path,"calibration_personality_scores_train.csv")):
                print("loading calibration personality scores train.csv")
                candidate_scores=pd.read_csv(os.path.join(result_temp_path,"calibration_personality_scores_train.csv"))
            else:
                print("caculating calibration personality scores train")
                candidate_scores=personality_scorer(candidate_generation,device,result_temp_path)

            if os.path.exists(os.path.join(result_temp_path,"calibration_personality_ranked_train.csv")):
                candidate_output=pd.read_csv(os.path.join(result_temp_path,"calibration_personality_ranked_train.csv"))
            else:
                ground_truth_scores=pd.read_csv("./dataset/retrieved_dataset/train.csv")
                candidate_output=personality_loss(candidate_scores,ground_truth_scores,result_temp_path)
        
            dataset = load_dataset("empathetic_dialogues")
            conversation_train=pd.read_csv("./dataset/retrieved_dataset/train.csv")
            conversation_valid=pd.read_csv("./dataset/retrieved_dataset/valid.csv")
            conversation_test=pd.read_csv("./dataset/retrieved_dataset/test.csv")
            
            print("train shape: ", len(conversation_train))
            print("validation shape: ", len(conversation_valid))
            print("test shape: ", len(conversation_test))   
            
            conversation_train_new = conversation_train.loc[conversation_train.index.repeat(args.top_can_num+1)]
            conversation_train_new.reset_index(drop=True, inplace=True)
            can_response = candidate_output["inputs"].tolist()
            
            total_length = len(conversation_train_new)
            all_indices = np.arange(total_length)
            update_indices = np.delete(all_indices, np.arange(0, total_length, args.top_can_num + 1))
            conversation_train_new.loc[update_indices, "response"] = can_response[:len(update_indices)]  
            
            encoded_train_data = convert_and_map(conversation_train_new, encode_example,tokenizer, roberta_tokenizer, data_type="train")
            encoded_validation_data = convert_and_map(conversation_valid, encode_example,tokenizer, roberta_tokenizer, data_type="validation")
            encoded_test_data = convert_and_map(conversation_test, encode_example, tokenizer, roberta_tokenizer, data_type="test")
            
            encoded_dataset = DatasetDict({
                'train': encoded_train_data,
                'validation': encoded_validation_data,
                'test': encoded_test_data
            })

            encoded_dataset.save_to_disk(dataset_path)
            del dataset, encoded_dataset
        
        dataset = load_from_disk(f"{data_path}/empathetic_dataset_retrieval4_calibrate") 
        print("train shape: ", len(dataset["train"]))
        print("validation shape: ", len(dataset["validation"]))
        print("test shape: ", len(dataset["test"]))
    
    print(f"data loaded from {data_path}")
    # data_columns=["input_ids","attention_mask","labels","context_ids","context_attention_mask"]
    data_columns = ['input_ids', 'attention_mask', 'labels']
    
    if args.stylizeEncoder:       
        if args.style=="personality":
            data_columns += ["context_ids","context_attention_mask","personality_ids","personality_attention_mask"]
            model=StyleRGPT2_P.from_pretrained("microsoft/DialoGPT-small",roberta_tokenizer=roberta_tokenizer,args=args) 
        
        elif args.style=="empathy":
            data_columns += ["context_ids","context_attention_mask","empathy_ids","empathy_attention_mask"]
            model=StyleRGPT2_E.from_pretrained("microsoft/DialoGPT-small",roberta_tokenizer=roberta_tokenizer,args=args)
       
        elif args.style=="both":
            data_columns += ["context_ids","context_attention_mask","personality_ids","personality_attention_mask","empathy_ids","empathy_attention_mask"] 
            model = StyleRGPT2_calibrate.from_pretrained("microsoft/DialoGPT-small",roberta_tokenizer=roberta_tokenizer,args=args)
    
        elif args.style=="context":  
            data_columns += ["context_ids","context_attention_mask"]
            model = RGPT2.from_pretrained("microsoft/DialoGPT-small",roberta_tokenizer=roberta_tokenizer,args=args)
        else:
            # model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small")
            model=CustomGPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small")
        wandb.init(project="Empathetic_GPT2_calibration", name=save_variable)
    else:
        model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small")
        wandb.init(project="Empathetic_GPT2", name=save_variable)
        
    # model = torch.nn.DataParallel(model)
    # model = model.to(device)
        
    dataset.set_format(
    type="torch",
    columns=data_columns)   

    data_collator = DataCollator(
        tokenizer, 
        roberta_tokenizer,
        model,
    )

    training_args = TrainingArguments(
        output_dir=model_output_path, 
        overwrite_output_dir=True,
        num_train_epochs=15, 
        per_device_train_batch_size=args.batch_size,  
        per_device_eval_batch_size=64, 
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        label_smoothing_factor=0.0,
        warmup_steps=args.warmup,
        weight_decay=0.01, 
        fp16=True,
        logging_dir=args.log_path,
        evaluation_strategy="steps",
        logging_first_step=False,
        logging_steps=100,     # eval_steps is default to this value 100
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=1,
        do_eval=True, 
        do_predict=False,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        disable_tqdm=not args.tqdm,
        group_by_length=False,
        length_column_name="length",
        dataloader_num_workers=8,
        remove_unused_columns=False,
        # resume_from_checkpoint=os.path.join(model_output_path, "checkpoint-5000"),
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        lm_loss_weight=args.lm_weight,
        per_loss_weight=args.per_weight,
        true_weight=args.true_weight,
        top_can_num=args.top_can_num,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"# Params: {pytorch_total_params}")

    print(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    
    print("model output path: ", model_output_path)
    print("start training")
    print(trainer.train())
    # assert trainer.state.global_step > 0, "The training didn't resume from the checkpoint."
    # print("Training resumed from checkpoint at global step:", trainer.state.global_step)
    print("finish training")
    print("evaluate on validation set")
    print(trainer.evaluate())
    print("evaluate on test set")
    print(trainer.evaluate(dataset["test"]))
    print("generate responses")
    generated_responses=generation(trainer, dataset["test"], result_output_path)
    custom_evalutions(generated_responses,result_output_path)
    eval(generated_responses, result_output_path)
    

def generation(trainer, dataset, result_path):
    set_seed(42)

    generated = generate_responses(
        trainer,
        dataset,
        max_length=128, 
        num_return_sequences=1, 
        do_sample=True,
        top_p=0.8,
        top_k=0,
        temperature=0.7,
        # num_beams=3,
        # no_repeat_ngram_size=3,
        disable_tqdm=not tqdm,
    ) 
    
    generation=pd.DataFrame(generated)
    generation.to_csv(os.path.join(result_path, "generated_p0.8_t0.7_calibration.csv"))
    
    return generation

if __name__ == "__main__":
    main()
    
    
    
    
    
        














    
    
    
    
    
    
    
    
    
    
 