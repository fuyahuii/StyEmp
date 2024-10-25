import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import argparse
import pandas as pd
from tqdm.auto import tqdm

from utils.generator_utils import encode_example, DataCollator, generate_responses, generate_responses_calibrate
import wandb
from datasets import load_dataset, load_from_disk, Dataset,DatasetDict
from models.dialogpt import Roberta_GPT2 as RGPT2
from models.dialogpt import StyleRoberta_GPT2 as StyleRGPT2
from models.dialogpt import StyleRoberta_GPT2_Personality as StyleRGPT2_P
from models.dialogpt import StyleRoberta_GPT2_Empathy as StyleRGPT2_E
from models.dialogpt import CustomGPT2LMHeadModel
from models.dialogpt import StyleRoberta_GPT2_calibrate as StyleRGPT2_calibrate
from custom_eval.inference import custom_evalutions
from eval import eval

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

            
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    parser= argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="path to the dataset", default="calibration/empathetic_dataset_retrieval_calibrate")
    parser.add_argument('--model_path', type=str, help="path to save trained model after calibration", default="calibration/output/model")
    parser.add_argument('--result_input_path', type=str, help="path to obtain generated responses before calibration", default="calibration/input")
    parser.add_argument('--result_output_path', type=str, help="path to save generated responses after calibration", default="calibration/output/result")
    parser.add_argument('--temp_path', type=str, help="path to save temporary files", default="calibration/output/temp")
    parser.add_argument('--log_path', type=str, help="path to save log", default="calibration/output/log")
    
    parser.add_argument("--stylizeEncoder", default=True, help="whether to use stylizeEncoder")
    parser.add_argument("--style",type=str, default="both", help="choose from [personality, empathy, both, context,none]")
    parser.add_argument('--personality_reinforcement', default=True, help="whether to use personality reinforcement")
    parser.add_argument('--addcontext', default='False', help="whether to add context slots")
    parser.add_argument('--concontext', default='True', help="whether to add context embeddings")
    parser.add_argument('--diffencoder', default='True', help="whether to use different encoder for style and context")
    
    parser.add_argument('--num_candidate', type=int, default=5)
    parser.add_argument('--batch_size',type=int, default=64, help="batch size in MgPE+DialoGPT")
    parser.add_argument('--bz',type=int, default=96, help="batch size in personality reinforcement module") #96 4
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
    
    data_path=args.data_path
    temp_path=args.temp_path
    model_path=args.model_path
    log_path=args.log_path
    result_path=args.result_output_path
    stylizeEncoder=args.stylizeEncoder
    style=args.style
 
    save_variable=f"style_{args.style}_batch_size_{args.batch_size}_lr_{args.lr}_warmup_{args.warmup}_speaker_{args.speaker_slots}_empathy_{args.empathy_slots}_addcontext_{args.addcontext}_concontext_{args.concontext}_diffencoder_{args.diffencoder}" 
    data_path=os.path.join(args.data_path,save_variable,f"num_candidate_{args.num_candidate}")
    
    save_variable=save_variable+f"num_candidate_{args.num_candidate}_true_weight_{args.true_weight}_per_weight_{args.per_weight}_lm_weight_{args.lm_weight}_bz_{args.bz}" #after calibration
    model_path=os.path.join(model_path,save_variable)
    log_path=os.path.join(log_path,save_variable)
    result_path=os.path.join(result_path,save_variable)
   
    
    #create directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    
    # if args.calibration:   
    #     calibrate_input=os.path.join(calibrate_input,save_variable,f"num_candidate_{args.num_candidate}")
    #     if not os.path.exists(calibrate_input):
    #         os.makedirs(calibrate_input)
        
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

    if stylizeEncoder:
        dataset = load_from_disk(f"{data_path}")

    else:  
        dataset = load_from_disk("dataset/empathetic_dataset")
        
    print(f"data loaded from {data_path}")
    data_columns=["input_ids","attention_mask","labels"]

    
    if stylizeEncoder:       
        if style=="personality":
            data_columns += ["context_ids","context_attention_mask","personality_ids","personality_attention_mask"]
            model = StyleRGPT2_P.from_pretrained(os.path.join(model_path,'checkpoint-2000'),local_files_only=True,roberta_tokenizer=roberta_tokenizer,args=args)
           
        elif style=="empathy":
            data_columns += ["context_ids","context_attention_mask","empathy_ids","empathy_attention_mask"]
            model = StyleRGPT2_E.from_pretrained(os.path.join(model_path,'checkpoint-2000'),local_files_only=True,roberta_tokenizer=roberta_tokenizer,args=args)

        elif style=="both":
            data_columns += ["context_ids","context_attention_mask","personality_ids","personality_attention_mask","empathy_ids","empathy_attention_mask"] 
            model = StyleRGPT2_calibrate.from_pretrained(os.path.join(model_path,'checkpoint-8000'),local_files_only=True,roberta_tokenizer=roberta_tokenizer,args=args)
    
        elif style=="context":  
            data_columns += ["context_ids","context_attention_mask"]
            model = RGPT2.from_pretrained(os.path.join(model_path,'checkpoint-2000'),local_files_only=True,roberta_tokenizer=roberta_tokenizer,args=args)  
        else:
            model=CustomGPT2LMHeadModel.from_pretrained(os.path.join(model_path,'checkpoint-2000'),local_files_only=True)
            print("model loaded")
    else:
        model = GPT2LMHeadModel.from_pretrained("output/model/checkpoint-2000",local_files_only=True)
       
        
    dataset.set_format(
    type="torch",
    columns=data_columns)   
    # max_style_num = args.num_ref
    # max_semantic_num = args.num_ref
    data_collator = DataCollator(
        tokenizer, 
        roberta_tokenizer,
        model,
        # max_style_num=max_style_num,
        # max_semantic_num=max_semantic_num,
    )

    training_args = TrainingArguments(
        save_strategy="no",
        output_dir=temp_path, 
        overwrite_output_dir=True,
        num_train_epochs=10, 
        per_device_train_batch_size=args.batch_size,  
        per_device_eval_batch_size=64, 
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        label_smoothing_factor=0.0,
        warmup_steps=0,
        weight_decay=0.01, 
        fp16=False,
        logging_dir=None,
        evaluation_strategy="steps",
        logging_first_step=False,
        logging_steps=2000,    # eval_steps is default to this value
        save_steps=2000,
        save_total_limit=1,
        do_eval=True, 
        do_predict=False,
        metric_for_best_model="loss",
        load_best_model_at_end=False,
        greater_is_better=False,
        disable_tqdm=not args.tqdm,
        group_by_length=False,
        length_column_name="length",
        dataloader_num_workers=16,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )
      
    generated_responses=generation(trainer, dataset["test"], result_path,args)
    custom_evalutions(generated_responses,result_path)
    eval(generated_responses, result_path)
    
        
def generation(trainer, dataset, result_path,args):
    set_seed(42)
    
    generated = generate_responses(
        trainer,
        dataset,
        max_length=128, 
        num_return_sequences=1, 
        do_sample=True,
        top_p=0.9,
        top_k=0,
        temperature=0.8,
        # num_beams=3,
        # no_repeat_ngram_size=3,
        disable_tqdm=not tqdm,
    )
    generation=pd.DataFrame(generated)
    generation.to_csv(os.path.join(result_path, "generated_p0.9_t0.8.csv"))
    
    return generation


if __name__ == "__main__":
    main()
    