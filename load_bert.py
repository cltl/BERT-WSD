import pandas as pd

import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel

bert_model_variation = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(bert_model_variation, do_lower_case=True)
model = BertModel.from_pretrained(bert_model_variation)

print(model)
