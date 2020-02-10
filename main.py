#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import argparse

from os.path import join
from datetime import datetime
from utils import get_logger, set_seed
from model import Model
from ptb import build_datasets, create_dataloader


def main():
    # directory for training outputs
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())

    # required parameters
    parser = argparse.ArgumentParser()

    parser.add_argument("--d_model", default=256, type=int,
                        help="Word embedding size and Tranformer hidden size.")
    parser.add_argument("--N", default=3, type=int,
                        help="Transformer stack number.")
    parser.add_argument("--head_num", default=8, type=int,
                        help="Head number.")
    parser.add_argument("--d_ff", default=512, type=int,
                        help="Linear layer size.")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate.")

    parser.add_argument("--hidden_size", default=16, type=int,
                        help="Batch size for training.")
    parser.add_argument("--num_embeddings", default=512, type=int,
                        help="Batch size for training.")
    parser.add_argument("--commitment_cost", default=0.25, type=float,
                        help="Batch size for training.")

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length (including eos token).")
    parser.add_argument("--seed", default=610, type=int,
                        help="Random seed.")
    parser.add_argument("--num_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--scheduler", default='warmup', type=str,
                        help="Which type of scheduler to use.")
    parser.add_argument("--optimizer", default='adamw', type=str,
                        help="Which type of optimizer to use.")
    parser.add_argument('--write_summary', default=True, type=bool,
                        help="If write summary into tensorboard.")
    parser.add_argument('--fp16', action='store_true', default=False,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=5, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_cuda", default=False, type=bool,
                        help="Do not use cuda.")
    parser.add_argument("--ignore_pad_idx", action='store_true', default=False,
                        help="Do not commpute loss for padding index.")

    parser.add_argument("--data_dir", default='data/', type=str,
                        help="data directory.")
    parser.add_argument("--output_dir", default=output_dir, type=str,
                        help="output directory for model, log file and summary.")
    parser.add_argument("--log_path", default=join(output_dir, "log.txt"), type=str,
                        help="Path to log.txt.")
    parser.add_argument("--summary_path", default=join(output_dir, "summary"), type=str,
                        help="Path to summary file.")
    parser.add_argument("--save_dir", default=join(output_dir, "model/"), type=str,
                        help="where to load pre-trained model.")

    args = parser.parse_args()

    # set random seed
    set_seed(args.seed)

    # create model output directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # create logger
    args.logger = get_logger(args.log_path)

    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.logger.info("- device: {}, n_gpu: {}".format(args.device, args.n_gpu))

    # update batch size
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # create dataset
    datasets = build_datasets(args)
    data_loaders = create_dataloader(datasets, args)

    # update args
    args.vocab_size = datasets['train'].vocab_size
    args.eos_idx = datasets['train'].eos_idx
    args.pad_idx = datasets['train'].pad_idx

    # build model
    args.logger.info("Building model...")
    model = Model(args)

    # training
    args.logger.info("Start training !!!")
    model.fit(data_loaders['train'], data_loaders['valid'])


if __name__ == '__main__':
    main()
