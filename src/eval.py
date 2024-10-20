import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from ignite.metrics import RougeL
from nltk.translate.meteor_score import meteor_score
from bleurt import score as bleurt_score
from bert_score import score
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from sentence_transformers import SentenceTransformer,util
import csv
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

def calc_distinct_n(n, candidates, print_score: bool = True):
    ngram_counts = {}
    total = 0
    candidates = [word_tokenize(candidate) for candidate in candidates]

    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i : i + n])
            ngram_counts[ngram] = 1
            total += 1

    score = len(ngram_counts) / (total + 1e-16)

    if print_score:
        print(f"***** Distinct-{n}: {score*100:.2f} *****")
        
    return score


def calc_distinct(candidates, print_score: bool = True):
    scores = []
    for i in range(2):
        score = calc_distinct_n(i + 1, candidates, print_score)
        scores.append(score)

    return scores

def corpus_meteor(refs, cands):
    scores = []
    for ref, cand in zip(refs, cands):
        scores.append(meteor_score(ref, cand))
    return np.mean(scores)

def compute_bleu(ref_list, hyp_list, no_overlap=False):
    ret_scores = {}
    
    # refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    refs = {idx: [lines.strip()] for (idx, lines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    # print(refs)
    # print(hyps)
    
    if not no_overlap:
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.6f" % (m, sc))
                    ret_scores[m] = sc
            else:
                print("%s: %0.6f" % (method, score))
                ret_scores[method] = score
            if isinstance(scorer, Meteor):
                scorer.close()
        del scorers

    return ret_scores

def style_similarity(refs, cands):
    style_model = SentenceTransformer('AnnaWegmann/Style-Embedding')

    style_embeddings_refs = style_model.encode(refs,batch_size=2048,device='cuda')
    style_embeddings_cands= style_model.encode(cands,batch_size=2048,device='cuda')
    print(style_embeddings_refs.shape,style_embeddings_cands.shape)
    style_similarity = util.pytorch_cos_sim(style_embeddings_refs, style_embeddings_cands)
    
    return style_similarity

def bleurt(label_texts, generated_texts,filename):
    scorer = bleurt_score.BleurtScorer('./BLEURT-20') # please download the bleurt checkpoint from https://github.com/google-research/bleurt
    bleurt_result = scorer.score(references=label_texts, candidates=generated_texts)
    print(filename)
    print(f"BELURT: {np.mean(bleurt_result)*100:.2f}") 
    del scorer
    
    return bleurt_result
    
    
def eval(generations, results_path):    

    filename=results_path.split('/')[-1]
    print("filename",filename)

    generated_text=generations['generated'].tolist()
    label_text=generations['label'].tolist()
    
    print(len(generated_text),len(label_text))
    print(generated_text[0])
    print(label_text[0])  

    filtered_pairs = [(gen, label) for gen, label in zip(generated_text, label_text) if isinstance(gen, str)]
    generated_texts, label_texts = zip(*filtered_pairs)
    
    generated_texts = [item.lower() for item in generated_texts]
    label_texts = [item.lower() for item in label_texts]
    
    print(len(generated_text),len(label_text))

    # BERTScore
    # print("Computing BERTScore...")
    P, R, F = score(generated_texts, label_texts,model_type="microsoft/deberta-large-mnli", lang="en")
    print(f"BERTScore: {float(F.mean())*100:.2f}")
 
    # BLEU
    combined_txts = [word_tokenize(item) for item in tqdm(generated_texts, disable=True)]
    combined_refs = [[word_tokenize(item)] for item in tqdm(label_texts, disable=True)]

    bleu_1 = corpus_bleu(combined_refs, combined_txts, weights=(1.0,))
    bleu_2 = corpus_bleu(combined_refs, combined_txts, weights=(0.5, 0.5))
    bleu_3 = corpus_bleu(combined_refs, combined_txts, weights=(1./3, 1./3, 1./3))
    bleu_4 = corpus_bleu(combined_refs, combined_txts, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"BLEU-(1|2|3|4): {bleu_1*100:.2f} | {bleu_2*100:.2f} | {bleu_3*100:.2f} | {bleu_4*100:.2f}")
    
    # style similarity
    style_sim = style_similarity(label_texts, generated_texts)
    print(f"Style similarity: {style_sim.mean()*100:.2f}")
   
    # rouge-l
    rouge_l = RougeL()
    rouge_l.update((combined_txts, combined_refs))
    rouge_l_score = rouge_l.compute()
    print(f"ROUGE-L: {rouge_l_score['Rouge-L-F']*100:.2f}")

    # meteor
    meteor = corpus_meteor(combined_refs, combined_txts)
    print(f"METEOR: {meteor*100:.2f}")
    
    distinct_score = calc_distinct(generated_texts)
    print(f"Distinct-1: {distinct_score[0]*100:.2f}")
    print(f"Distinct-2: {distinct_score[1]*100:.2f}")
    
    # BLEURT
    bleurt_result=0
    bleurt_result=bleurt(label_texts, generated_texts,filename)

    output_file=open('output/result/eval_result.csv','a')
    # output_file=open('calibration/output/result/eval_result.csv','a')
    csv_writer = csv.writer(output_file, delimiter=',', quotechar='"')
    
    csv_writer.writerow([filename])
    csv_writer.writerow(['BLEU-1','BLEU-2','BLEU-3','BLEU-4','ROUGE-L','METEOR','BERTScore','BLEURT','Distinct-1','Distinct-2','Style similarity'])
    csv_writer.writerow([format(bleu_1*100, '.2f'), format(bleu_2*100, '.2f'), format(bleu_3*100, '.2f'), format(bleu_4*100, '.2f'), \
                        format(rouge_l_score['Rouge-L-F']*100, '.2f'), format(meteor*100, '.2f'), format(float(F.mean())*100, '.2f'), format(np.mean(bleurt_result)*100, '.2f'), \
                        format(distinct_score[0]*100, '.2f'), format(distinct_score[1]*100, '.2f'), format(style_sim.mean()*100, '.2f')])