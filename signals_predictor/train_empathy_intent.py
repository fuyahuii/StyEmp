import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import pandas as pd
import random
import argparse
import tqdm
import torch.nn as nn
import torch
from transformers import AdamW,get_linear_schedule_with_warmup, AutoTokenizer, AutoModel	
from models import Empathy_Intent_Encoder

# from evaluation_utils import *
from sklearn.metrics import accuracy_score,accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix

import logging

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", default="./dataset_intents", help="path to data folder, choose from [ ./dataset_intents; ./dataset_empathy]")
parser.add_argument("--seed_val", default=12, type=int, help="Seed value")
parser.add_argument("--model_output", default="./outputs", help="path to save model")
parser.add_argument("--log_path", default="./intent_logs", help="path to save logs, choose from [./empathy_logs, ./intent_logs]")
# parser.add_argument("--result_path", default="./intents_results", help="path to save results, choose from [./empathy_results, ./intent_results]")
parser.add_argument("--task", default="intent", help="task name from [intent,emotion_react,interpretations,explorations]")
parser.add_argument("--test_only", action="store_true", help="whether only do test")

parser.add_argument("--lr", default=2e-5, type=float, help="learning rate")
parser.add_argument("--dropout", default=0.1, type=float, help="dropout")
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--max_len", default=512, type=int, help="max length of input")
parser.add_argument("--epochs", default=7, type=int, help="number of the max epochs")
parser.add_argument("--pretrain_model", default="studio-ousia/luke-base", help="base model, choose from [studio-ousia/luke-base, roberta-base]")
parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
parser.add_argument("--early_stop", default=3, type=int, help="patience for early stopping")

args = parser.parse_args()

variant=f"{args.task}_{args.pretrain_model}_epochs{args.epochs}_lr{args.lr}_dropout{args.dropout}_warmup{args.warmup_steps}_seed{args.seed_val}"
log_path=os.path.join(args.log_path,variant)
model_output_path=os.path.join(args.model_output,variant)

if not os.path.exists(model_output_path):
    os.makedirs(model_output_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
# if not os.path.exists(args.result_path):
#     os.makedirs(args.result_path)

logging.basicConfig(filename=os.path.join(log_path,'log.txt'), level=logging.INFO)
logging.info("\n")
logging.info(args)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data=data
        self.tokenizer=tokenizer
        self.max_len=max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample=self.data[index]
        encoding=self.tokenizer(sample["input"],return_tensors='pt',max_length=self.max_len,truncation=True,padding='max_length')
       
        return {
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten(),
            'label':torch.tensor(sample["label"],dtype=torch.long),
            'input':sample["input"],
        }
        
def cal_loss(model,batch,device,args):
    input_ids=batch['input_ids'].to(device)
    attention_mask=batch['attention_mask'].to(device)
    label=batch['label'].to(device)
    
    logits=model(input_ids=input_ids,attention_mask=attention_mask)
    loss_fct = nn.CrossEntropyLoss()
    
    loss=loss_fct(logits, label)

    return loss,logits

def train_valid(model,train_dataloader,valid_dataloader,device,args):
    
    optimizer = AdamW(model.parameters(),
				  lr = args.lr,
				  eps = 1e-8
				)
    total_steps = len(train_dataloader) * args.epochs
    num_batch = len(train_dataloader)

    print('total_steps =', total_steps)
    print('num_batch =', num_batch)
    print("=============================================")

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    random.seed(args.seed_val)
    np.random.seed(args.seed_val)
    torch.manual_seed(args.seed_val)
    torch.cuda.manual_seed_all(args.seed_val)

    for epoch_i in range(args.epochs):
        total_train_loss = 0
        model.train()

        print("Epoch: {}".format(epoch_i))
        logging.info("Epoch: {}".format(epoch_i))
        
        for batch in tqdm.tqdm(train_dataloader,desc=f"training"):
            optimizer.zero_grad()        
            loss,_=cal_loss(model,batch,device,args)
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
   
        avg_train_loss = total_train_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))
        logging.info("Average train loss: {}".format(avg_train_loss))
        
        print("Start evaluating on validation data...")
        logging.info("Start evaluating on validation data...")
        total_eval_loss = 0
        model.eval()
        
        for batch in valid_dataloader:
            with torch.no_grad():
                loss,logits=cal_loss(model,batch,device,args)
                total_eval_loss += loss.item()
        avg_val_loss = total_eval_loss / len(valid_dataloader)
        print("Average validation loss: {}".format(avg_val_loss))
        logging.info("Average validation loss: {}".format(avg_val_loss))
        
        
    torch.save(model.state_dict(), os.path.join(model_output_path,"model.pt"))
       

def evaluate(model,test_dataloader,device,args):
    print("Start evaluating on test data...")
    logging.info("Start evaluating on test data...")
    
    model.eval()
    total_test_loss = 0
    predictions=[]
    trues=[]
    for batch in test_dataloader:
        with torch.no_grad():
            loss,logits=cal_loss(model,batch,device,args)
            total_test_loss += loss.item()
            predictions+=torch.argmax(logits,dim=1).tolist()
            trues+=batch['label'].tolist()
    avg_test_loss = total_test_loss / len(test_dataloader)
    print("Average test loss: {}".format(avg_test_loss))
    logging.info("Average test loss: {}".format(avg_test_loss))
    
    accuracy=accuracy_score(trues,predictions)
    balanced_accuracy=balanced_accuracy_score(trues,predictions)
    f1_micro=f1_score(trues,predictions,average='micro')
    f1_macro=f1_score(trues,predictions,average='macro')
    f1_weighted=f1_score(trues,predictions,average='weighted')

    logging.info("Accuracy: {}".format(accuracy))
    logging.info("Balanced Accuracy: {}".format(balanced_accuracy))
    logging.info("F1_micro: {}".format(f1_micro))
    logging.info("F1_macro: {}".format(f1_macro))
    logging.info("F1_weighted: {}".format(f1_weighted))
    
   
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
        
    logging.info("device: {}".format(device))
    logging.info("Loading data...")
    
    # load data
    train_data,valid_data,test_data=[],[],[]
    if args.task=="intent":
        
        df_train=open(os.path.join(args.data_path,'train.txt')).readlines()
        df_valid=open(os.path.join(args.data_path,'valid.txt')).readlines()
        df_test=open(os.path.join(args.data_path,'test.txt')).readlines()

        train={'input':[],'label':[]}
        valid={'input':[],'label':[]}
        test={'input':[],'label':[]}

        for line in df_train:
            label=line.strip().split('<SEP>')[0]
            data=line.strip().split('<SEP>')[1]
            if int(label)>31:
                train['input'].append(data)
                train['label'].append(int(label)-32)

        for line in df_valid:
            label=line.strip().split('<SEP>')[0]
            data=line.strip().split('<SEP>')[1]
            if int(label)>31:
                valid['input'].append(data)
                valid['label'].append(int(label)-32)
        
        for line in df_test:
            label=line.strip().split('<SEP>')[0]
            data=line.strip().split('<SEP>')[1]
            if int(label)>31:
                test['input'].append(data)
                test['label'].append(int(label)-32)

        for i in range(len(train['input'])):
            train_data.append({'input':train['input'][i],'label':train['label'][i]})
        for i in range(len(valid['input'])):
            valid_data.append({'input':valid['input'][i],'label':valid['label'][i]})
        for i in range(len(test['input'])):
            test_data.append({'input':test['input'][i],'label':test['label'][i]})

        logging.info("total size of task {}: {}".format(args.task,len(train_data)+len(valid_data)+len(test_data)))
        logging.info("train size of task {}: {}".format(args.task,len(train_data)))
        logging.info("valid size of task {}: {}".format(args.task,len(valid_data)))
        logging.info("test size of task {}: {}".format(args.task,len(test_data)))
        num_labels=len(set(train['label']+valid['label']+test['label']))
    
    else:
        data=os.path.join(args.data_path,args.task+".csv")
        df=pd.read_csv(data,delimiter=',')
        length=len(df)

        df['level']=df['level'].apply(lambda x: 1 if x>=1 else 0)
        print(df['level'].value_counts())
        logging.info("number of level 0|1: {}".format(df['level'].value_counts()))

        df_train=df[:int(length*0.75)]
        df_valid=df[int(length*0.75):int(length*0.8)]
        df_test=df[int(length*0.8):]
        
        train={'input':df_train['response_post'].tolist(),'label':df_train['level'].tolist()}
        valid={'input':df_valid['response_post'].tolist(),'label':df_valid['level'].tolist()}
        test={'input':df_test['response_post'].tolist(),'label':df_test['level'].tolist()}
        
        for i in range(len(train['input'])):
            train_data.append({'input':train['input'][i],'label':train['label'][i]})
        for i in range(len(valid['input'])):
            valid_data.append({'input':valid['input'][i],'label':valid['label'][i]})
        for i in range(len(test['input'])):
            test_data.append({'input':test['input'][i],'label':test['label'][i]})

        logging.info("train size of task {}: {}".format(args.task,len(train_data)))
        logging.info("valid size of task {}: {}".format(args.task,len(valid_data)))
        logging.info("test size of task {}: {}".format(args.task,len(test_data)))
        num_labels=len(set(df['level'].tolist()))
    
    print("num_labels: {}".format(num_labels))
    logging.info("num_labels: {}".format(num_labels))    
    
    tokenizer=AutoTokenizer.from_pretrained(args.pretrain_model)
    train_dataset=Dataset(train_data,tokenizer,args.max_len)
    valid_dataset=Dataset(valid_data,tokenizer,args.max_len)
    test_dataset=Dataset(test_data,tokenizer,args.max_len)
    
    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    valid_dataloader=torch.utils.data.DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=False)
    test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False)
    
    base_model=AutoModel.from_pretrained(args.pretrain_model)
    model=Empathy_Intent_Encoder(base_model=base_model,hidden_dropout_prob=args.dropout,num_labels=num_labels)
    model.to(device)
    
    if args.test_only:
        model.load_state_dict(torch.load(os.path.join(model_output_path,'model.pt')))
        evaluate(model,test_dataloader,device,args)
    else:
        logging.info("Start training...")
        train_valid(model,train_dataloader,valid_dataloader,device,args)
        logging.info("Start testing...")
        model.load_state_dict(torch.load(os.path.join(model_output_path,'model.pt')))
        evaluate(model,test_dataloader,device,args)
        
if __name__=="__main__":
    main()
    

    
   
    
    
