#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import nltk
import random
import logging
import tensorflow as tf
from glob import glob
from tensorflow.keras.utils import Progbar
sys.path.append("bert")
import sentencepiece as spm
import argparse
import re


regex_tokenizer = nltk.RegexpTokenizer("\w+")
regex_tokenizer = nltk.RegexpTokenizer("[^_\n]+")

# Remove none characheter/emoji UTF-8
def cleannnorm_tweets(text):
    reg = re.compile(u'['
          u'\U0001F300-\U0001F64F'
          u'\U0001F680-\U0001F6FF'
          u'\u2600-\u26FF\u2700-\u27BF]',
          re.UNICODE)
    new_text = reg.sub(r' \g<0> ',text)
    new_text = re.sub("[^\wu\U0001F300-\U0001F64Fu\U0001F680-\U0001F6FFu\u2600-\u26FF\u2700-\u27BF]+"," ",new_text)
    new_text = re.sub("[_]"," ",new_text)
    return new_text

# Remove non-UTF codes
def normalize_text(text):
  # lowercase text
  text = str(text).lower()
  # remove non-UTF
  text = text.encode("utf-8", "ignore").decode()
  # remove punktuation symbols
  #text = " ".join(regex_tokenizer.tokenize(text))
  text = cleannnorm_tweets(text)

  return text

# Count Lines for Progbar
def count_lines(filename):
  count = 0
  with open(filename) as fi:
      for line in fi:
          count += 1
  return count

# Load vocab file
def read_sentencepiece_vocab(filepath):
  voc = []
  with open(filepath, encoding='utf-8') as fi:
    for line in fi:
      voc.append(line.split("\t")[0])
  # skip the first <unk> token
  voc = voc[1:]
  return voc


def parse_sentencepiece_token(token):
    if token.startswith("▁"):
        return token[1:]
    else:
        return "##" + token


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--infile','-i', help='training file')
  parser.add_argument('--vocab_size','-s', help='testing file')

  args = parser.parse_args()

  inputFile = args.infile
  vocab_size  = int(args.vocab_size)

  total_lines = count_lines(inputFile)
  bar = Progbar(total_lines)

  MODEL_PREFIX = "tokenizer" #@param {type: "string"}
  VOC_SIZE = vocab_size #@param {type:"integer"}
  SUBSAMPLE_SIZE = 5600000 #total_lines #128000 #@param {type:"integer"}
  NUM_PLACEHOLDERS = 256 #@param {type:"integer"}
  PRC_DATA_FPATH = inputFile+'.proc'

  SPM_COMMAND = ('--input={} --model_prefix={} '
                 '--vocab_size={} --input_sentence_size={} '
                 '--shuffle_input_sentence=true ' 
                 '--bos_id=-1 --eos_id=-1').format(
                 PRC_DATA_FPATH, MODEL_PREFIX, 
                 VOC_SIZE - NUM_PLACEHOLDERS, SUBSAMPLE_SIZE)
  print("Runing: "+SPM_COMMAND)
  spm.SentencePieceTrainer.Train(SPM_COMMAND)

  snt_vocab = read_sentencepiece_vocab("{}.vocab".format(MODEL_PREFIX))
  print("Learnt vocab size: {}".format(len(snt_vocab)))
  print("Sample tokens: {}".format(random.sample(snt_vocab, 10)))

  print("Split "+PRC_DATA_FPATH+" into shards....")
  os.system('split -a 4 -l 256000 -d '+PRC_DATA_FPATH+' ./shards/shard_')

  bert_vocab = list(map(parse_sentencepiece_token, snt_vocab))

  ctrl_symbols = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
  bert_vocab = ctrl_symbols + bert_vocab

  bert_vocab += ["[UNUSED_{}]".format(i) for i in range(VOC_SIZE - len(bert_vocab))]
  print(len(bert_vocab))

  VOC_FNAME = inputFile+".vocab" #@param {type:"string"}

  with open(VOC_FNAME, "w") as fo:
      for token in bert_vocab:
          fo.write(token+"\n")

if __name__ == "__main__":
  main()





