import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import shutil
import argparse
import pandas as pd
from tqdm.auto import tqdm

from utils.generator_utils import encode_example, DataCollator, generate_responses
import wandb
from datasets import load_dataset, load_from_disk, Dataset,DatasetDict
from models.dialogpt import Roberta_GPT2 as RGPT2
from models.dialogpt import StyleRoberta_GPT2 as StyleRGPT2
from models.dialogpt import StyleRoberta_GPT2_Personality as StyleRGPT2_P
from models.dialogpt import StyleRoberta_GPT2_Empathy as StyleRGPT2_E
from eval import eval
from custom_eval.inference import custom_evalutions

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

    
def convert_to_conversation(dataset,filename):
    dataset=dataset.to_pandas()
    conversations = dataset["conv_id"].unique().tolist()
    data = []
    total_dialogue=[]
    total_utterance=0
    print(f"Loading data from {filename}")
    for conv_id in tqdm(conversations):
        conv = dataset.query(f'conv_id == "{conv_id}"').sort_values("utterance_idx")
        context = []
        speakers=[]
        for idx, utterance in enumerate(conv.iterrows()):
            utterance = utterance[1]
            curr_utterance = utterance["utterance"].replace("_comma_", ",")
            curr_speaker = utterance["speaker_idx"]
            speakers.append(curr_speaker)
            if idx % 2 == 1:
                # curr_utterance = "[SPK2] " + curr_utterance
                data.append({
                    "emotion": utterance["context"],
                    "context": "; ".join(context[:idx]),
                    "response": curr_utterance,
                    "user_id":speakers[idx-1],
                    "responder_id": utterance["speaker_idx"],
                    })
            # else:
            #     curr_utterance = "[SPK1] " + curr_utterance
            context.append(curr_utterance)
        total_dialogue.append(conv)
        total_utterance+=len(conv)
    for i in range (5):
        print(filename+" sample data: ", data[i])    
    print(filename+" totall dilaogues: ",len(total_dialogue))
    print(filename+" dialogue utterances:",total_utterance)
    print(filename+" totall data: ",len(data))  
    return data 

def convert_data(dataset_subset,cancatenated_speaker_history, personality, empathy_explanation, retrieval_conversation):
    big5_e, mbti_intro, mbti_thinking=personality
    for i in range(len(dataset_subset)):
        dataset_subset[i]["cancatenated_speaker_history"]=cancatenated_speaker_history[i].replace('[SEP]','; ')
        dataset_subset[i]["big5_e"]=big5_e[i]
        dataset_subset[i]["mbti_intro"]=mbti_intro[i]
        dataset_subset[i]["mbti_thinking"]=mbti_thinking[i]
        dataset_subset[i]["empathy_explanation"]=empathy_explanation[i]+" The response is: "+retrieval_conversation[i]["response"]
    print("sample data: ", dataset_subset[0])
    return dataset_subset

def convert_and_map(dataset_subset, encode_func,cancatenated_speaker_history, personality, empathy_explanation,retrieval_conversation, tokenizer, roberta_tokenizer):
    print("data shape: ", len(dataset_subset))
    converted_data = convert_data(dataset_subset,cancatenated_speaker_history, personality, empathy_explanation, retrieval_conversation)
    converted_dataset = Dataset.from_pandas(pd.DataFrame(converted_data))
    return converted_dataset.map(encode_func, fn_kwargs={"tokenizer": tokenizer, "roberta_tokenizer": roberta_tokenizer}, num_proc=16)

def convert(dataset_subset, filename, encode_func, tokenizer, roberta_tokenizer):
    print(f"{filename} data shape: ", dataset_subset.shape)
    converted_data = convert_to_conversation(dataset_subset, filename)
    converted_dataset = Dataset.from_pandas(pd.DataFrame(converted_data))
    return converted_dataset.map(encode_func, fn_kwargs={"tokenizer": tokenizer, "roberta_tokenizer": roberta_tokenizer}, num_proc=16)

def retrieval_process(dataset,args):
    
    conversation_train= convert_to_conversation(dataset['train'], "train")
    conversation_dev= convert_to_conversation(dataset['validation'], "validation")
    conversation_test= convert_to_conversation(dataset['test'], "test")
    
    if not os.path.exists(args.temp_path+"/speaker_history_train.pkl"):      
        conversation_train_new,cancatenated_speaker_history_train, big5_e_train, mbti_intro_train, mbti_thinking_train =speaker_history_retrieval(conversation_train,conversation_train,"train",args.temp_path)
        conversation_dev_new,cancatenated_speaker_history_dev, big5_e_dev, mbti_intro_dev, mbti_thinking_dev =speaker_history_retrieval(conversation_train,conversation_dev,"dev",args.temp_path)
        conversation_test_new,cancatenated_speaker_history_test, big5_e_test, mbti_intro_test, mbti_thinking_test=speaker_history_retrieval(conversation_train,conversation_test,"test",args.temp_path)

        with open(os.path.join(args.temp_path, "speaker_history_train.pkl"), "wb") as f:
            pickle.dump([conversation_train_new,cancatenated_speaker_history_train,  big5_e_train, mbti_intro_train, mbti_thinking_train], f)
        with open(os.path.join(args.temp_path, "speaker_history_dev.pkl"), "wb") as f:
            pickle.dump([conversation_dev_new,cancatenated_speaker_history_dev, big5_e_dev, mbti_intro_dev, mbti_thinking_dev], f)
        with open(os.path.join(args.temp_path, "speaker_history_test.pkl"), "wb") as f:
            pickle.dump([conversation_test_new,cancatenated_speaker_history_test, big5_e_test, mbti_intro_test, mbti_thinking_test], f)
    else:
        print("loading speaker history from pickle file")
        with open(os.path.join(args.temp_path, "speaker_history_train.pkl"), "rb") as f:
            conversation_train_new,cancatenated_speaker_history_train, big5_e_train, mbti_intro_train, mbti_thinking_train = pickle.load(f)
        with open(os.path.join(args.temp_path, "speaker_history_dev.pkl"), "rb") as f:
            conversation_dev_new,cancatenated_speaker_history_dev, big5_e_dev, mbti_intro_dev, mbti_thinking_dev = pickle.load(f)
        with open(os.path.join(args.temp_path, "speaker_history_test.pkl"), "rb") as f:
            conversation_test_new,cancatenated_speaker_history_test, big5_e_test, mbti_intro_test, mbti_thinking_test = pickle.load(f)
            
    retrieved_conversation_dev=conversation_retrieval(conversation_train,conversation_dev_new)
    retrieved_conversation_test=conversation_retrieval(conversation_train,conversation_test_new)

        
    if not os.path.exists(args.temp_path+"/empathy_explanation_test.pkl"):
        empathy_explanation_train=empathy_explanation(conversation_train)
        with open(os.path.join(args.temp_path, "empathy_explanation_train.pkl"), "wb") as f:
            pickle.dump(empathy_explanation_train, f)
       
        empathy_explanation_dev=empathy_explanation(retrieved_conversation_dev)
        with open(os.path.join(args.temp_path, "empathy_explanation_dev.pkl"), "wb") as f:
            pickle.dump(empathy_explanation_dev, f)
            
        empathy_explanation_test=empathy_explanation(retrieved_conversation_test)
        with open(os.path.join(args.temp_path, "empathy_explanation_test.pkl"), "wb") as f:
            pickle.dump(empathy_explanation_test, f)
            
    else:
        print("loading empathy explanation from pickle file")
        with open(os.path.join(args.temp_path, "empathy_explanation_train.pkl"), "rb") as f:
            empathy_explanation_train = pickle.load(f)
        with open(os.path.join(args.temp_path, "empathy_explanation_dev.pkl"), "rb") as f:
            empathy_explanation_dev = pickle.load(f)
        with open(os.path.join(args.temp_path, "empathy_explanation_test.pkl"), "rb") as f:
            empathy_explanation_test = pickle.load(f)
            
    conversation_train_updated = [
        {**item, "empathy_explanation": empathy_explanation_train[i],"cancatenated_speaker_history": cancatenated_speaker_history_train[i], 
        "big5_e": big5_e_train[i], "mbti_intro": mbti_intro_train[i], "mbti_thinking": mbti_thinking_train[i]}
        for i, item in enumerate(conversation_train)
    ]
    pd.DataFrame(conversation_train_updated).to_csv(os.path.join(args.data_path, "retrieved_dataset/dataset4/train.csv"))

    conversation_dev_new_updated = [
        {**item, "retrieved_context": retrieved_conversation_dev[i]["context"], 
        "retrieved_response": retrieved_conversation_dev[i]["response"],
        "empathy_explanation": empathy_explanation_dev[i],
        "retrieved_response_speaker": retrieved_conversation_dev[i]["responder_id"],
        "cancatenated_speaker_history": cancatenated_speaker_history_dev[i], 
        "big5_e": big5_e_dev[i], "mbti_intro": mbti_intro_dev[i], "mbti_thinking": mbti_thinking_dev[i]}
        for i, item in enumerate(conversation_dev_new)
    ]
    pd.DataFrame(conversation_dev_new_updated).to_csv(os.path.join(args.data_path, "retrieved_dataset/dataset4/valid.csv"))
    
    conversation_test_new_updated = [
        {**item, "retrieved_context": retrieved_conversation_test[i]["context"], 
        "retrieved_response": retrieved_conversation_test[i]["response"],
        "empathy_explanation": empathy_explanation_test[i],
        "retrieved_response_speaker": retrieved_conversation_test[i]["responder_id"],
        "cancatenated_speaker_history": cancatenated_speaker_history_test[i], 
        "big5_e": big5_e_test[i], "mbti_intro": mbti_intro_test[i], "mbti_thinking": mbti_thinking_test[i]}
        for i, item in enumerate(conversation_test_new)
    ]
    pd.DataFrame(conversation_test_new_updated).to_csv(os.path.join(args.data_path, "retrieved_dataset/dataset4/test.csv"))  

    print("train data shape: ", len(conversation_train_updated))
    print("dev data shape: ", len(conversation_dev_new_updated))
    print("test data shape: ", len(conversation_test_new_updated))
    
    personality_train=[big5_e_train, mbti_intro_train, mbti_thinking_train]
    personality_dev=[big5_e_dev, mbti_intro_dev, mbti_thinking_dev]
    personality_test=[big5_e_test, mbti_intro_test, mbti_thinking_test]
    
    return  conversation_train, cancatenated_speaker_history_train,personality_train,empathy_explanation_train,\
            conversation_dev_new,   cancatenated_speaker_history_dev,personality_dev,empathy_explanation_dev,retrieved_conversation_dev,\
            conversation_test_new,cancatenated_speaker_history_test,personality_test, empathy_explanation_test, retrieved_conversation_test

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    parser= argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="path to the dataset", default="dataset")
    parser.add_argument("--temp_path", type=str, help="path to save temporary data files", default="dataset/temp/dataset3")
    parser.add_argument('--model_path', type=str, help="path to save trained model", default="output/model/dataset3")
    parser.add_argument('--log_path', type=str, help="path to save log", default="output/log/dataset3")
    parser.add_argument('--result_path', type=str, help="path to save result", default="output/result/dataset3")
    parser.add_argument("--stylizeEncoder", default=True, help="whether to use stylizeEncoder")
    parser.add_argument("--style",type=str, default="both", help="choose from [personality, empathy, both, context,none]")
    parser.add_argument('--personality_reinforcement', action='store_true', help="whether to use personality reinforcement")
    parser.add_argument('--addcontext', default='False', help="whether to add context slots")
    parser.add_argument('--concontext', default='True', help="whether to add context embeddings")
    parser.add_argument('--diffencoder', default='False', help="whether to use different encoder for style and context")
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--tqdm', default=True,help="whether to use tqdm")
    parser.add_argument('--speaker_slots', type=int, default=25, help="number of speaker slots")
    parser.add_argument('--empathy_slots', type=int, default=25, help="number of empathy slots")
    
    args=parser.parse_args()
    print(args)
    
    data_path=args.data_path
    temp_path=args.temp_path
    model_path=args.model_path
    log_path=args.log_path
    result_path=args.result_path
    stylizeEncoder=args.stylizeEncoder
    style=args.style

    save_variable=f"style_{args.style}_batch_size_{args.batch_size}_lr_{args.lr}_warmup_{args.warmup}_speaker_{args.speaker_slots}_empathy_{args.empathy_slots}_dataset3_addcontext_{args.addcontext}_concontext_{args.concontext}_diffencoder_{args.diffencoder}" 
    #addcontext is false and con means concatenate style and context by dimension 2, only style slots
    #addcontext is true means concatenate style and context by dimension 1, means both style and context slots.
    # save_variable=f"style_{args.style}_batch_size_{args.batch_size}_lr_{args.lr}_warmup_{args.warmup}_dataset3_addcontext"
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
        
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    
    #check whether [SPK1] and [SPK2] are in the vocabulary
    # for add_token in ["[SPK1]","[SPK2]"]:
    #     if add_token not in tokenizer.get_vocab():
    #         tokenizer.add_tokens(add_token)
    #         roberta_tokenizer.add_tokens(add_token)

    if stylizeEncoder:
        dataset_path = os.path.join(data_path, "empathetic_dataset_retrieval3")
 
        if local_rank <= 0 and not os.path.exists(dataset_path):
            print(f"tokenizing data in {data_path} ...")
            
            dataset = load_dataset("empathetic_dialogues")
            conversation_train, cancatenated_speaker_history_train,personality_train,empathy_explanation_train,\
            conversation_dev,   cancatenated_speaker_history_dev,personality_dev,empathy_explanation_dev,retrieved_conversation_dev,\
            conversation_test,cancatenated_speaker_history_test,personality_test, empathy_explanation_test,retrieved_conversation_test=retrieval_process(dataset,args)

            encoded_train_data = convert_and_map(conversation_train, encode_example,cancatenated_speaker_history_train, personality_train, empathy_explanation_train, conversation_train,tokenizer, roberta_tokenizer)
            encoded_validation_data = convert_and_map(conversation_dev, encode_example, cancatenated_speaker_history_dev,personality_dev, empathy_explanation_dev, retrieved_conversation_dev,tokenizer, roberta_tokenizer)
            encoded_test_data = convert_and_map(conversation_test, encode_example,cancatenated_speaker_history_test,personality_test, empathy_explanation_test, retrieved_conversation_test, tokenizer, roberta_tokenizer)
            
            encoded_dataset = DatasetDict({
                'train': encoded_train_data,
                'validation': encoded_validation_data,
                'test': encoded_test_data
            })

            encoded_dataset.save_to_disk(dataset_path)
            del dataset, encoded_dataset
        
        dataset = load_from_disk(f"{data_path}/empathetic_dataset_retrieval3")
   

    else:
        dataset_path = os.path.join(data_path, "empathetic_dataset")
        
        if local_rank <= 0 and not os.path.exists(dataset_path):
            print(f"tokenizing data in {data_path} ...")
            dataset = load_dataset("empathetic_dialogues")
            encoded_train_data = convert(dataset['train'], "train", encode_example, tokenizer, roberta_tokenizer)
            encoded_validation_data = convert(dataset['validation'], "validation", encode_example, tokenizer, roberta_tokenizer)
            encoded_test_data = convert(dataset['test'], "test", encode_example, tokenizer, roberta_tokenizer)

            encoded_dataset = DatasetDict({
                'train': encoded_train_data,
                'validation': encoded_validation_data,
                'test': encoded_test_data
            })
            encoded_dataset.save_to_disk(dataset_path)
            del dataset, encoded_dataset
           
        dataset = load_from_disk(f"{data_path}/empathetic_dataset")
        
    print(f"data loaded from {data_path}")
    # data_columns=["input_ids","attention_mask","labels","context_ids","context_attention_mask"]
    data_columns = ['input_ids', 'attention_mask', 'labels']
    
    if stylizeEncoder:       
        if style=="personality":
            data_columns += ["context_ids","context_attention_mask","personality_ids","personality_attention_mask"]
            model=StyleRGPT2_P.from_pretrained("microsoft/DialoGPT-small",roberta_tokenizer=roberta_tokenizer,args=args) 
        
        elif style=="empathy":
            data_columns += ["context_ids","context_attention_mask","empathy_ids","empathy_attention_mask"]
            model=StyleRGPT2_E.from_pretrained("microsoft/DialoGPT-small",roberta_tokenizer=roberta_tokenizer,args=args)
       
        elif style=="both":
            data_columns += ["context_ids","context_attention_mask","personality_ids","personality_attention_mask","empathy_ids","empathy_attention_mask"] 
            model = StyleRGPT2.from_pretrained("microsoft/DialoGPT-small",roberta_tokenizer=roberta_tokenizer,args=args)
    
        elif style=="context":  
            data_columns += ["context_ids","context_attention_mask"]
            model = RGPT2.from_pretrained("microsoft/DialoGPT-small",roberta_tokenizer=roberta_tokenizer)
        else:
            model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small")
        wandb.init(project="Empathetic_GPT2", name=save_variable)
    else:
        model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small")
        wandb.init(project="Empathetic_GPT2", name=save_variable)
           
    # model.resize_token_embeddings(len(tokenizer))
    
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
        output_dir=model_path, 
        overwrite_output_dir=True,
        num_train_epochs=15, 
        per_device_train_batch_size=args.batch_size,  
        per_device_eval_batch_size=args.batch_size, 
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        label_smoothing_factor=0.0,
        warmup_steps=args.warmup,
        weight_decay=0.01, 
        fp16=True,
        logging_dir=log_path,
        evaluation_strategy="steps",
        logging_first_step=False,
        logging_steps=100,    # eval_steps is default to this value
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

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"# Params: {pytorch_total_params}")

    print(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    print("start training")
    print(trainer.train())
    print("finish training")
    print("evaluate on validation set")
    print(trainer.evaluate())
    print("evaluate on test set")
    print(trainer.evaluate(dataset["test"]))
    print("generate responses")
    generated_responses=generation(trainer, dataset["test"], result_path)
    eval(generated_responses, result_path)
    custom_evalutions(generated_responses,result_path)
    
    
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
    generation.to_csv(os.path.join(result_path, "generated_p0.8_t0.7.csv"))
    
    return generation

if __name__ == "__main__":
    main()
    