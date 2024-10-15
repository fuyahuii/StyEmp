**This repository is currently being updated and is not yet complete.**

The official codes for the paper: Yahui Fu, Chenhui Chu, Tatsuya Kawahara. "[StyEmp: Stylizing Empathetic Response Generation via Multi-Grained Prefix Encoder and Personality Reinforcement](https://arxiv.org/pdf/2408.02271)." SIGDIAL 2024.
If you find this repository or paper useful, please kindly cite our paper:
```
@inproceedings{fu2024styemp,
  title={StyEmp: Stylizing Empathetic Response Generation via Multi-Grained Prefix Encoder and Personality Reinforcement},
  author={Fu, Yahui and Chu, Chenhui and Kawahara, Tatsuya},
  booktitle={Proceedings of the 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue},
  pages={172--185},
  year={2024}
}
```
### **Abstract**
<p style="text-align: justify;">
Recent approaches for empathetic response
generation mainly focus on emotional resonance and user understanding, without considering the system’s personality. Consistent personality is evident in real human expression
and is important for creating trustworthy systems. To address this problem, we propose
StyEmp, which aims to stylize the empathetic
response generation with a consistent personality. Specifically, it incorporates a multi-grained
prefix mechanism designed to capture the intricate relationship between a system’s personality and its empathetic expressions. Furthermore, we introduce a personality reinforcement
module that leverages contrastive learning to
calibrate the generation model, ensuring that
responses are both empathetic and reflective
of a distinct personality. Automatic and human evaluations on the EMPATHETICDIALOGUES benchmark show that StyEmp outperforms competitive baselines in terms of both
empathy and personality expressions.
</p>

<div align="center">
  <img src="./Figs/example.png" alt="Description" width="450"/>
</div>


### **Model Architecture**
<div align="center">
  <img src="./Figs/arch.png" alt="Model Architecture" width="450"/>
</div>

### **Preparing Environment**
```
mamba env create -f env.yml -n styemp
conda activate styemp
```
### **Personality, ECM, and Intent Predictors Preparation**
```
python signals_predictor/train_personality.py
python signals_predictor/rain_empathy_intent.py
```
The best checkpoints for predictors can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1DnkYM8XA1b9o-b8eN1fHk96_Va0tkoxI), which will be further used as the custom evaluation metrics, please put them into the folder src/custom_eval/pretrained_signals.

### **Multi-grained Prefix Encoder**
#### **Training**
```
python src/train_generator.py
```
The best checkpoints can be downloaded from [here](https://drive.google.com/drive/u/0/folders/10vlibYEYHXvqjey9mZP9qBYGzV5q_2w5).
#### **Inference**
```
python src/inference.py
```
To use the Multi-grained Prefix Encoder with DialoGPT for response generation only, set the `calibration` argument to `False`.
Setting `calibration` to `True` will generate multiple candidate responses, as demonstrated in the Personality Reinforcement module, which is needed for the following training.

### **Multi-grained Prefix Encoder with Personality Reinforcement**
#### **Training**
```
python src/calibrate_generator.py
```
The best checkpoint can be downloaded from [here](https://drive.google.com/drive/u/0/folders/15SaRe9akp1ONGFUkhp-7LKXTXYWnR5r7).
#### **Inference**
```
python src/inference_calibration.py
```


