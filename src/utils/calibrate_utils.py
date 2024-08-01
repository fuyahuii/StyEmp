import numpy as np
import pandas as pd
import os
import torch
from utils.empathy_intent_personality import PersonalityClassifier

candidate_num = 5

def personality_scorer(inputs,device,result_path):
    personality_classifier=PersonalityClassifier(device)
    personality_big5_e, personality_mbti_intro, personality_mbti_thinking =personality_classifier.predict_personality(inputs)
    
    personality_big5_e=np.array(personality_big5_e,dtype=np.float32).flatten().tolist()
    personality_mbti_intro=np.array(personality_mbti_intro,dtype=np.float32).flatten().tolist()
    personality_mbti_thinking=np.array(personality_mbti_thinking,dtype=np.float32).flatten().tolist()
    
    scores={"inputs": inputs,"big5_e":personality_big5_e,"mbti_intro":personality_mbti_intro,"mbti_thinking":personality_mbti_thinking}
    personality_scores=pd.DataFrame(scores)
    personality_scores.to_csv(os.path.join(result_path,"calibration_personality_scores_train.csv"))
    
    return personality_scores

def personality_loss(candidate_scores,ground_truth_scores,result_path):
    big5_e=np.array(ground_truth_scores["big5_e"].astype(float).tolist())
    mbti_intro=np.array(ground_truth_scores["mbti_intro"].astype(float).tolist())
    mbti_thinking=np.array(ground_truth_scores["mbti_thinking"].astype(float).tolist())
    
    candidates=candidate_scores["inputs"].tolist()
    candidates_big5_e=np.array(candidate_scores["big5_e"].astype(float).tolist())
    candidates_mbti_intro=np.array(candidate_scores["mbti_intro"].astype(float).tolist())
    candidates_mbti_thinking=np.array(candidate_scores["mbti_thinking"].astype(float).tolist())
    
    candidates_output={"inputs":[], "mse_loss":[]}
    print("big5_e: ", len(big5_e))
    print("candidates_big5_e: ", len(candidates_big5_e))
    
    for i in range(len(big5_e)):
        temp_candidates = []
        temp_mse_losses = []
        temp_mae_losses = []    
        
        for j in range(candidate_num):
            idx = candidate_num*i + j
            mse_big5_e = np.square(candidates_big5_e[idx] - big5_e[i]).mean()
            mse_mbti_intro = np.square(candidates_mbti_intro[idx] - mbti_intro[i]).mean()
            mse_mbti_thinking = np.square(candidates_mbti_thinking[idx] - mbti_thinking[i]).mean()
            
            mae_big5_e = np.abs(candidates_big5_e[idx] - big5_e[i]).mean()
            mae_mbti_intro = np.abs(candidates_mbti_intro[idx] - mbti_intro[i]).mean()
            mae_mbti_thinking = np.abs(candidates_mbti_thinking[idx] - mbti_thinking[i]).mean()
              
            # Calculate total loss
            mse_loss = mse_big5_e + mse_mbti_intro + mse_mbti_thinking
            mae_loss = mae_big5_e + mae_mbti_intro + mae_mbti_thinking
            
            temp_candidates.append(candidates[idx]) 
            temp_mse_losses.append(mse_loss)
            temp_mae_losses.append(mae_loss)
    
        # Sort candidates based on their loss from low to high
        candidates_sorted = sorted(zip(temp_candidates,temp_mse_losses), key=lambda x: x[1])
    
        # Append the top 10 candidates to the output
        for j in range(candidate_num):
            candidates_output["inputs"].append(candidates_sorted[j][0])
            candidates_output["mse_loss"].append(candidates_sorted[j][1])
    
    candidates_output=pd.DataFrame(candidates_output)
    candidates_output.to_csv(os.path.join(result_path,"calibration_personality_ranked_train.csv"))
  
    return candidates_output

def RankingLoss(scores, true_weight):
    
    if true_weight > 0:
        candidate_scores=scores[:,1:]
        true_score=scores[:,0]
        ones = torch.ones_like(candidate_scores)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(candidate_scores, candidate_scores, ones)

        # candidate loss
        n = candidate_scores.size(1)
        for i in range(n-1):
            for j in range(i+1, n):
                pos_score = candidate_scores[:, i].unsqueeze(-1)
                neg_score = candidate_scores[:, j].unsqueeze(-1)
                ones = torch.ones_like(pos_score)
                loss_func = torch.nn.MarginRankingLoss(0.001*(j-i))
                loss = loss_func(pos_score, neg_score, ones)
                TotalLoss += loss
        TotalLoss = TotalLoss / (n * (n - 1) / 2)
        
        # gold response loss
        pos_score = true_score.unsqueeze(-1).expand_as(candidate_scores)
        neg_score = candidate_scores
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones_like(pos_score)
        loss_func = torch.nn.MarginRankingLoss(0)
        TotalLoss += true_weight * loss_func(pos_score, neg_score, ones)
    
    else:    
        ones = torch.ones_like(scores)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(scores, scores, ones)
    
        n = scores.size(1)
        for i in range(n-1):
            for j in range(i+1, n):
                pos_score = scores[:, i].unsqueeze(-1)
                neg_score = scores[:, j].unsqueeze(-1)
                ones = torch.ones_like(pos_score)
                loss_func = torch.nn.MarginRankingLoss(0.001*(j-i))
                loss = loss_func(pos_score, neg_score, ones)
                TotalLoss += loss
        TotalLoss = TotalLoss / (n * (n - 1) / 2)
     
    return TotalLoss