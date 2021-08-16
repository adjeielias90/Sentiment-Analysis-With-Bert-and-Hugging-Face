import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import json
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
# from collections import defaultdict
# from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class_names = ['negative', 'neutral', 'positive']
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LEN = 160
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
# review_text = "I couldn't figure this out. Worst app ever!"


class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)


model = SentimentClassifier(len(class_names))
model.load_state_dict(torch.load('./best_model_state.bin', map_location=torch.device('cpu')))
# model = model.to(device)




def getReview(review_text):
  encoded_review = tokenizer.encode_plus(
    review_text,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
  )

  input_ids = encoded_review['input_ids']
  attention_mask = encoded_review['attention_mask']

  output = model(input_ids, attention_mask)
  _, prediction = torch.max(output, dim=1)
  return prediction




def get_prediction_as_json(review_text):
  # predicted_sentiment = getReview(review_text)
  # true_sentiment_class = class_names[predicted_sentiment]
  
  prediction = getReview(str(review_text))
  prediction = class_names[prediction]

  response = {'review': str(review_text), 'prediction': str(prediction)}
  json_dump = json.dumps(response)
  json_obj = json.loads(json_dump)
  return json_obj

# prediction = get_prediction_as_json(review_text)
# print(prediction)































# encoded_review = tokenizer.encode_plus(
#   review_text,
#   max_length=MAX_LEN,
#   add_special_tokens=True,
#   return_token_type_ids=False,
#   pad_to_max_length=True,
#   return_attention_mask=True,
#   return_tensors='pt',
# )

# input_ids = encoded_review['input_ids']
# attention_mask = encoded_review['attention_mask']

# output = model(input_ids, attention_mask)
# _, prediction = torch.max(output, dim=1)

# print(f'Review text: {review_text}')
# print(f'Sentiment  : {class_names[prediction]}')


# def getReview(review_text):
#   encoded_review = tokenizer.encode_plus(
#     review_text,
#     max_length=MAX_LEN,
#     add_special_tokens=True,
#     return_token_type_ids=False,
#     pad_to_max_length=True,
#     return_attention_mask=True,
#     return_tensors='pt',
#   )

#   input_ids = encoded_review['input_ids']
#   attention_mask = encoded_review['attention_mask']

#   output = model(input_ids, attention_mask)
#   _, prediction = torch.max(output, dim=1)
#   return prediction




