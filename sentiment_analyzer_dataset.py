import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
# from collections import defaultdict
from textwrap import wrap
import json
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# from sentiment_analyzer import getReview

# if __name__ == '__main__':
#   torch.multiprocessing.freeze_support()


class_names = ['negative', 'neutral', 'positive']
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LEN = 160
BATCH_SIZE = 16
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
df_test = pd.read_csv("./df_test.csv")


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



def get_review_from_id(id):
  sample_review = df_test.iloc[int(id)]
  true_sentiment = sample_review['sentiment']
  sentiment = sample_review['content'] 
  return sentiment, true_sentiment

def get_prediction_as_json_with_id(id):
  review, true_sentiment = get_review_from_id(id)
  true_sentiment_class = class_names[true_sentiment]
  prediction = getReview(review)
  prediction = class_names[prediction]

  response = {'review': str(review), 'true_sentiment': str(true_sentiment_class), 'prediction': str(prediction)}
  json_dump = json.dumps(response)
  json_obj = json.loads(json_dump)
  return json_obj

prediction = get_prediction_as_json_with_id(15)
print(prediction)