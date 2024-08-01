import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from datasets import load_dataset
from models.retrieval_explanation import conversation_retrieval,speaker_history_retrieval,empathy_explanation
from utils.empathy_intent_personality import EmotionClassifier, PersonalityClassifier, EmpathyClassifier
import pandas as pd
import sys

dataset = load_dataset("empathetic_dialogues")
train=dataset['train']
valid=dataset['validation']
test=dataset['test']

# concatenate the train, valid and test set
train_df=pd.DataFrame(train)
valid_df=pd.DataFrame(valid)
test_df=pd.DataFrame(test)
all_df=pd.concat([train_df, valid_df, test_df])
print(all_df.shape)

unique_speaker=set(all_df['speaker_idx'])
print("unique speaker in all three sets: ", len(unique_speaker))

data=pd.read_csv('src/speaker_utterances.csv')
speaker_data=[]
utterances=[]
speaker_utterances=[]
for speaker in unique_speaker:
    speaker_data.append(data[data['speaker']==str(speaker)])   
    for i in range (min(20,len(speaker_data[-1]))):
        utterances.append(speaker_data[-1].iloc[i]['utterances'])
    speaker_utterances.append(utterances)

device='cuda'
personality_classifier=PersonalityClassifier(device)

for i in range(len(speaker_utterances)):
    selected_speaker_histories=' '.join(speaker_utterances[i])
    print(selected_speaker_histories)
    personality_big5_e, personality_mbti_intro, personality_mbti_thinking =personality_classifier.predict_personality(selected_speaker_histories)
    print(personality_big5_e, personality_mbti_intro, personality_mbti_thinking)
    sys.exit()


