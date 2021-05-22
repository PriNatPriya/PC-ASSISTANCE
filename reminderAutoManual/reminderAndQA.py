# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:19:17 2021

@author: Dev
"""

import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

model = BertForQuestionAnswering.from_pretrained(r"F:\myWork\pcControl\QandA\preTrainnedModel")
tokenizer = BertTokenizer.from_pretrained(r"F:\myWork\pcControl\QandA\preTrainnedModel")
# model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


def answer_question(question, answer_text):
    input_ids = tokenizer.encode(question, answer_text)
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids])) 
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokens[answer_start]
    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]
    print('Answer: "' + answer + '"')
    

file1 = open(r"F:\myWork\pcControl\QandA\reminderAutoManual\infoDB.txt","r")
data = file1.read()
question = "where is my ration card"
answer_question(question, data)