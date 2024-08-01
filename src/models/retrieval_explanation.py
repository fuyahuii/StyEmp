
from transformers import AutoModel
from sentence_transformers import SentenceTransformer,util
from utils.empathy_intent_personality import EmotionClassifier, PersonalityClassifier, EmpathyClassifier
import torch
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pickle
import os

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")
 
emotion_classifier=EmotionClassifier(device)
personality_classifier=PersonalityClassifier(device)
empathy_classifier=EmpathyClassifier(device)

def compute_embeddings_response(dataset):
    sematic_model = SentenceTransformer('all-MiniLM-L6-v2') #based on bert, use [SEP] and [CLS]
    style_model = SentenceTransformer('AnnaWegmann/Style-Embedding') #based on roberta, use <s> and </s>
    
    sentences = [data["response"] for data in dataset]
    sematic_embeddings = sematic_model.encode(sentences,batch_size=2048)
    style_embeddings = style_model.encode(sentences,batch_size=2048)
    emotion_embeddings = emotion_classifier.predict_emotion(sentences)
    print("compute_embeddings_response done")
    
    return [sematic_embeddings, style_embeddings, emotion_embeddings],sentences

def compute_embeddings_context(dataset):
    sematic_model = SentenceTransformer('all-MiniLM-L6-v2')
    style_model = SentenceTransformer('AnnaWegmann/Style-Embedding')
    
    sentences = [data["context"] for data in dataset]
    sematic_embeddings = sematic_model.encode(sentences,batch_size=1024)
    style_embeddings = style_model.encode(sentences,batch_size=1024)
    emotion_embeddings = emotion_classifier.predict_emotion(sentences)
    print("compute_embeddings_context done")
    
    return [sematic_embeddings, style_embeddings, emotion_embeddings],sentences

def similarity_retrieval(train_embeddings, test_embeddings):
    sematic_embeddings_train, style_embeddings_train, emotion_embeddings_train=train_embeddings
    sematic_embeddings_test, style_embeddings_test, emotion_embeddings_test=test_embeddings

    sematic_cosine_scores = util.pytorch_cos_sim(sematic_embeddings_test, sematic_embeddings_train)
    style_cosine_scores = util.pytorch_cos_sim(style_embeddings_test, style_embeddings_train)
    emotion_cosine_scores = util.pytorch_cos_sim(emotion_embeddings_test, emotion_embeddings_train)

    similarity_scores = sematic_cosine_scores +  style_cosine_scores + emotion_cosine_scores
    assert len(sematic_embeddings_test)==len(similarity_scores)
    
    return similarity_scores

def conversation_retrieval(train, test):    
    retrieved_conversation = []
    train_embeddings,_ = compute_embeddings_context(train)
    test_embeddings,_ = compute_embeddings_context(test)
    retrieved_responses_scores=similarity_retrieval(train_embeddings, test_embeddings)
    
    max_score_indices = np.argmax(retrieved_responses_scores, axis=1)
    #choose the dict from train dataset based on the max_score_indices
    retrieved_conversation.extend([{key: train[max_score_indices[i]][key] for key in train[max_score_indices[i]].keys()} for i in range(len(max_score_indices))])
    print("retrieved conversation size: ", len(retrieved_conversation))
    print("test size: ", len(test))
    print("max_score_indices size: ", len(max_score_indices))
    assert len(retrieved_conversation)==len(test)==len(max_score_indices)
    print("response_retrieval done")
    
    test_speaker=[data["responder_id"] for data in test]
    retrieved_speaker=[data["responder_id"] for data in retrieved_conversation]
    #accuracy of speaker retrieval
    accuracy_speaker_retrieval=sum([1 if test_speaker[i]==retrieved_speaker[i] else 0 for i in range(len(test_speaker))])/len(test_speaker)
    print("accuracy_speaker_retrieval", accuracy_speaker_retrieval)
    return retrieved_conversation
        
def speaker_history_retrieval(dataset_train, dataset_test,data_type,temp_path):
    selected_speaker_history=[]
    
    dataset_train = pd.DataFrame(dataset_train)
    grouped_history = dataset_train.groupby('responder_id')
    
    full_speaker_history = {
    speaker: history_df.to_dict('records')
    for speaker, history_df in grouped_history}

    # Randomly sample up to 20 responses for each speaker in dataset_test
    speaker_history,history_length, new_dataset_test= [],[],[]
    for data in dataset_test:
        speaker = data["responder_id"]
        if speaker in full_speaker_history:
            speaker_responses = full_speaker_history[speaker]
            sampled_responses = np.random.choice(speaker_responses, size=min(30, len(speaker_responses)), replace=False)
            speaker_history.append(sampled_responses.tolist())
            history_length.append(len(sampled_responses))
            new_dataset_test.append(data)

    print("Sampled speaker_history size: ", len(speaker_history))
    speaker_history = [item for sublist in speaker_history for item in sublist]
    print("Flattened speaker_history size: ", len(speaker_history))
    
    if os.path.exists(temp_path+"/speaker_history_embeddings_"+data_type+".pkl"):
        print("loading speaker_history_embeddings_"+data_type+".pkl")
        with open(temp_path+'/speaker_history_embeddings_'+data_type+".pkl", 'rb') as f:
            speaker_history_embeddings=pickle.load(f)
        speaker_history_embeddings_3d_semantic=speaker_history_embeddings["speaker_history_embeddings_3d_semantic"]
        speaker_history_embeddings_3d_style=speaker_history_embeddings["speaker_history_embeddings_3d_style"]
        speaker_history_embeddings_3d_emotion=speaker_history_embeddings["speaker_history_embeddings_3d_emotion"]
        history_responses=speaker_history_embeddings["history_responses"]
    
        with open(temp_path+'/test_embeddings_'+data_type+".pkl", 'rb') as f:
            test_embeddings=pickle.load(f)
        test_semantic_embeddings=test_embeddings["test_semantic_embeddings"]
        test_style_embeddings=test_embeddings["test_style_embeddings"]
        test_emotion_embeddings=test_embeddings["test_emotion_embeddings"]
        test_responses=test_embeddings["test_responses"]
    else:    
        print("computing speaker_history_embeddings_"+data_type+".pkl")
        speaker_history_embeddings,history_responses=compute_embeddings_response(speaker_history)
        test_embeddings,test_responses=compute_embeddings_response(dataset_test)
        test_semantic_embeddings, test_style_embeddings, test_emotion_embeddings=test_embeddings
        test_emotion_embeddings=np.array(test_emotion_embeddings,dtype=np.float32)
        
        speaker_history_embeddings_semantic, speaker_history_embeddings_style, speaker_history_embeddings_emotion=speaker_history_embeddings

        speaker_history_embeddings_3d_semantic=np.zeros((len(history_length),max(history_length),len(speaker_history_embeddings_semantic[0])),dtype=np.float32)
        speaker_history_embeddings_3d_style=np.zeros((len(history_length),max(history_length),len(speaker_history_embeddings_style[0])),dtype=np.float32)
        speaker_history_embeddings_3d_emotion=np.zeros((len(history_length),max(history_length), len(speaker_history_embeddings_emotion[0])),dtype=np.float32)
        
        print("speaker_history_embeddings_sematic size: ", len(speaker_history_embeddings_semantic), len(speaker_history_embeddings_semantic[0]))
        print("speaker_history_embeddings_style size: ", len(speaker_history_embeddings_style), len(speaker_history_embeddings_style[0]))
        print("speaker_history_embeddings_emotion size: ", len(speaker_history_embeddings_emotion), len(speaker_history_embeddings_emotion[0]))
        print("speaker_history_embeddings_3d_semantic size: ", speaker_history_embeddings_3d_semantic.shape)
        print("speaker_history_embeddings_3d_style size: ", speaker_history_embeddings_3d_style.shape)
        print("speaker_history_embeddings_3d_emotion size: ", speaker_history_embeddings_3d_emotion.shape)
        
        for i in range(len(history_length)):   
            speaker_history_embeddings_3d_semantic[i,:history_length[i]]=speaker_history_embeddings_semantic[:history_length[i]]
            speaker_history_embeddings_3d_style[i,:history_length[i]]=speaker_history_embeddings_style[:history_length[i]]
            speaker_history_embeddings_3d_emotion[i,:history_length[i]]=speaker_history_embeddings_emotion[:history_length[i]]
            
            speaker_history_embeddings_semantic=speaker_history_embeddings_semantic[history_length[i]:]
            speaker_history_embeddings_style=speaker_history_embeddings_style[history_length[i]:]
            speaker_history_embeddings_emotion=speaker_history_embeddings_emotion[history_length[i]:]
        print("speaker_history_embeddings_semantic", speaker_history_embeddings_semantic)
        
        test_semantic_embeddings=np.expand_dims(test_semantic_embeddings,axis=1)
        test_style_embeddings=np.expand_dims(test_style_embeddings,axis=1)
        test_emotion_embeddings=np.expand_dims(test_emotion_embeddings,axis=1)
        
        speaker_to_save={
            "speaker_history_embeddings_3d_semantic":speaker_history_embeddings_3d_semantic,
            "speaker_history_embeddings_3d_style":speaker_history_embeddings_3d_style,
            "speaker_history_embeddings_3d_emotion":speaker_history_embeddings_3d_emotion,
            "history_responses":history_responses,
        }
        with open(temp_path+'/speaker_history_embeddings_'+data_type+".pkl", 'wb') as f:
            pickle.dump(speaker_to_save, f)
        
        test_to_save={
            "test_semantic_embeddings":test_semantic_embeddings,
            "test_style_embeddings":test_style_embeddings,
            "test_emotion_embeddings":test_emotion_embeddings,
            "test_responses":test_responses,
        }
        with open(temp_path+'/test_embeddings_'+data_type+".pkl", 'wb') as f:
            pickle.dump(test_to_save, f)
        
    similarity_score_all=[]
    for i in range(len(speaker_history_embeddings_3d_semantic)):
        speaker_history_embeddings=[speaker_history_embeddings_3d_semantic[i],speaker_history_embeddings_3d_style[i],speaker_history_embeddings_3d_emotion[i]]
        test_embeddings=[test_semantic_embeddings[i],test_style_embeddings[i],test_emotion_embeddings[i]] 
        similarity_score=similarity_retrieval(speaker_history_embeddings,test_embeddings)
        # check whether the similarity_score is bigger than 0 or not
        similarity_score_all.append(similarity_score.detach().cpu().numpy().flatten().tolist())
        
    print("similarity_score size: ", np.array(similarity_score_all).shape)
        
    for i in range(len(similarity_score_all)):
        #sort the similarity_score_all from high to low, and take the min(history_length[i], 10) indices
        max_score_indices = np.argsort([abs(num) for num in similarity_score_all[i]])[::-1][:min(history_length[i], 10)]
        history_res = history_responses[:history_length[i]]  
        selected_speaker_history.append([history_res[idx] for idx in max_score_indices])
        history_responses=history_responses[history_length[i]:]
    print("selected_speaker_history size: ", len(selected_speaker_history))
    print("dataset test: ", len(dataset_test))
    print("new_dataset_test: ", len(new_dataset_test))
    
    assert len(selected_speaker_history)==len(new_dataset_test)
    print("speaker_history_retrieval done")
    
    selected_speaker_histories=[" [SEP] ".join(selected_speaker_history[i]).replace("[SPK2]","").replace("[SPK1]","") for i in range(len(selected_speaker_history))]
    print("selected_speaker_histories size", len(selected_speaker_histories))
   
    personality_big5_e, personality_mbti_intro, personality_mbti_thinking =personality_classifier.predict_personality(selected_speaker_histories)
    personality_big5_e=np.array(personality_big5_e,dtype=np.float32).flatten().tolist()
    personality_mbti_intro=np.array(personality_mbti_intro,dtype=np.float32).flatten().tolist()
    personality_mbti_thinking=np.array(personality_mbti_thinking,dtype=np.float32).flatten().tolist()
    
    print("personality big5_e size of "+data_type, len(personality_big5_e))

    # cancatenated_speaker_history=[]
    # for i in range(len(selected_speaker_history)):
    #     selected_speaker_history[i]=selected_speaker_history[i].astype(str).tolist()
    #     selected_speaker_history[i]=personality[i]+'\n'.join(selected_speaker_history[i])
    #     cancatenated_speaker_history.append(selected_speaker_history[i])
        
    # print("cancatenated_speaker_history: ", cancatenated_speaker_history[0])
    return new_dataset_test,selected_speaker_histories, personality_big5_e, personality_mbti_intro, personality_mbti_thinking
    
def empathy_explanation(dataset):
    empathy_explanation=[]
    Emotion_intents=["agreeing","acknowledging","encouraging","consoling","sympathizing","suggesting","questioning","wishing","neutral"]
    predictions_ER_g, predictions_IP_g, predictions_EX_g, predictions_EI_g=empathy_classifier.predict_empathy_intent([data["response"] for data in dataset])
    print("empathy_classifier done")
    
    for i in range(len(dataset)):
        text='Response contains '
        # if predictions_ER_g[i]==0:
        #     text+='No emotional react.'
        if predictions_ER_g[i]==1:
            text+='emotional reaction, '
        # if predictions_IP_g[i]==0:
        #     text+='No interpretation.'
        if predictions_IP_g[i]==1:
            text+='interpretation, '
        # if predictions_EX_g[i]==0:
        #     text+='No exploration.'
        if predictions_EX_g[i]==1:
            text+='exploration. '
        if predictions_ER_g[i]==0 and predictions_IP_g[i]==0 and predictions_EX_g[i]==0:
            text=''
        text+="Emotion Intent is "+Emotion_intents[predictions_EI_g[i]]+"."
        empathy_explanation.append(text)
    return empathy_explanation
    
        
        
      
        
      