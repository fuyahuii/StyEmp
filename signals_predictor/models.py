import torch.nn as nn
import torch
from transformers import AutoModel


class Empathy_Intent_Encoder(nn.Module):
    def __init__(self, base_model,hidden_dropout_prob,num_labels):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(base_model)
        self.empathy_regression= Empathy_Intent_Head(input_size = 768,num_labels=num_labels,dropout=hidden_dropout_prob)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state

        logits = self.empathy_regression(sequence_output[:, 0, :])

        return  logits
    
class Empathy_Intent_Head(nn.Module):
    def __init__(self, input_size,num_labels,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(input_size, input_size)
        self.out_proj = nn.Linear(input_size, num_labels)

    def forward(self, x, **kwargs): 
        x=self.dropout(x)
        x = torch.relu(self.dense(x))
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class PersonalityEncoder(nn.Module):

	def __init__(self, base_model,hidden_dropout_prob, num_labels,hidden_size,num_task):
		super().__init__()
		self.encoder=AutoModel.from_pretrained(base_model)
		self.personality_regression= PersonalityHead(input_size = 768,num_labels = num_labels,dropout=hidden_dropout_prob,hidden_size=hidden_size,num_task=num_task)
	
	def forward(self, input_ids, attention_mask):
		outputs = self.encoder(
			input_ids=input_ids,
			attention_mask=attention_mask
		)
		sequence_output = outputs.last_hidden_state

		logits = self.personality_regression(sequence_output[:, 0, :])

		return  logits

class PersonalityHead(nn.Module):
	"""Head for sentence-level classification/regression tasks."""

	def __init__(self, input_size, num_labels,dropout,hidden_size,num_task):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		self.dense = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(num_task)])
		self.out_proj = nn.ModuleList([nn.Linear(hidden_size, num_labels) for _ in range (num_task)])

	def forward(self, x, **kwargs): 
		x = [torch.relu(dense(x)) for dense in self.dense]
		x = [self.dropout(x) for x in x]
		x = [out_proj(output) for output, out_proj in zip(x, self.out_proj)]
		return x


