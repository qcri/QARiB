#!/bin/sh

python3 ../bert/create_pretraining_data.py --input_file=./shards/$1 --output_file=pretraining_data/$1.tfrecord --vocab_file=data.cleaned.txt.vocab --do_lower_case=True --max_predictions_per_seq=20 --max_seq_length=64 --masked_lm_prob=0.15 --random_seed=34 --dupe_factor=5



