#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common PyTorch model training structure.

   Author: Meng Cao
"""

import os
import torch
import torch.nn as nn
# import torch.optim as optim
# import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm
# from apex import amp
from transformers import WarmupLinearSchedule
from autoencoder import EmbeddingLayer, Encoder, Decoder, LinearSoftmax, EncoderDecoder, VectorQuantizer


class Model:
    """This class implements all model training and evluation methods.
    """
    def __init__(self, args):
        """Initialize the model.
        """
        self.args = args
        self.logger = args.logger

        # initialize model
        self.model = self._build_model()
        self.model.to(args.device)

        # create optimizer and criterion
        self.optimizer = self._get_optimizer(self.model.parameters())
        self.scheduler = self._get_scheduler(self.optimizer)
        self.criterion = self._get_criterion(pad_idx=args.pad_idx)

        # Amp: Automatic Mixed Precision
        if self.args.fp16:
            self.model, self.optimizer = amp.initialize(self.model,
                                                        self.optimizer,
                                                        opt_level=args.fp16_opt_level)
            self.logger.info("- Automatic Mixed Precision (AMP) is used.")
        else:
            self.logger.info("- NO Automatic Mixed Precision (AMP) available :/")

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            self.logger.info("- Let's use {} GPUs !".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        else:
            self.logger.info("- Train the model on single GPU :/")

        # tensorboard
        if args.write_summary:
            self.writer = SummaryWriter(self.args.summary_path)

    def _build_model(self):
        """Build auto-encoder model.
        """
        embed = EmbeddingLayer(self.args.d_model,
                               self.args.vocab_size,
                               dropout=self.args.dropout)
        encoder = Encoder(self.args.d_model,
                          self.args.N,
                          self.args.head_num,
                          self.args.d_ff,
                          dropout=self.args.dropout)
        decoder = Decoder(self.args.d_model,
                          self.args.N,
                          self.args.head_num,
                          self.args.d_ff,
                          dropout=self.args.dropout)
        linear_softmax = LinearSoftmax(self.args.d_model, self.args.vocab_size)
        vector_quantizer = VectorQuantizer(self.args.hidden_size, self.args.num_embeddings,
                                           self.args.commitment_cost)
        model = EncoderDecoder(embed, encoder, decoder, linear_softmax, vector_quantizer)

        return model

    def _initialize_params(self, model):
        """Initialize model parameters.
        """
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def _get_optimizer(self, optimizer_grouped_parameters):
        """Get optimizer for model training.
        """
        if self.args.optimizer == 'adamw':
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=self.args.learning_rate,
                              eps=self.args.adam_epsilon)
        elif self.args.optimizer == 'adam':
            optimizer = Adam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             eps=self.args.adam_epsilon)
        else:
            raise Exception("Unknown optimizer type!")

        return optimizer

    def _get_scheduler(self, optimizer):
        """Get scheduler for adjusting learning rate.
        """
        if self.args.scheduler == 'warmup':
            scheduler = WarmupLinearSchedule(optimizer,
                                             warmup_steps=self.args.warmup_steps,
                                             t_total=self.args.num_epochs)
        elif self.args.scheduler == 'exponential':
            scheduler = ExponentialLR(optimizer, 0.95)
        return scheduler

    def _get_criterion(self, pad_idx=None):
        """Implement loss function.
        """
        if self.args.ignore_pad_idx and pad_idx is not None:
            return nn.NLLLoss(ignore_index=pad_idx)
        else:
            self.logger.info("- WARNNING: no pad-index ignored during training!")
            return nn.NLLLoss()

    def load_weights(self, path):
        """Load pre-trained weights.
        """
        self.model.load_state_dict(torch.load(path))

    def save_model(self, save_dir):
        """Save the model.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir + 'checkpoint.pth.tar')
        torch.save(self.model.state_dict(), model_path)

    def _build_mask(self, inputs, pad_idx=0):
        """Build mask for input sequence.

        Args:
            inputs: [batch_size, seq_len]
        """
        en_mask = torch.ones_like(inputs, dtype=inputs.dtype, device=self.args.device)
        en_mask.masked_fill_(inputs == pad_idx, 0)

        return en_mask

    def loss_batch(self, inputs, labels, optimizer=None, step=None):
        """Compute loss and update model weights on a batch of data.

        Args:
            inputs: [batch_size, seq_len]
            labels: [batch_size, seq_len]

        Returns:
            loss: float
            log_probs: [batch_size, seq_len, vocab_size]
        """
        mask = self._build_mask(inputs, self.args.pad_idx)
        log_probs, vq_vae_loss = self.model(inputs, mask)  # outputs: [N, S, vocab_size]
        print(log_probs.shape)
        print(vq_vae_loss.shape)
        print(vq_vae_loss)
        assert 1 == 0
        loss = self.criterion(log_probs.transpose(1, 2), labels)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if optimizer is not None:
            loss.backward()  # compute gradients

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.max_grad_norm)
                optimizer.step()  # update model parameters
                optimizer.zero_grad()  # clean all gradients

        return loss.item(), log_probs.detach()

    def train_epoch(self, train_dataloader, optimizer, epoch):
        """Train the model for one single epoch.
        """
        self.model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        train_loss = 0.0
        for i, batch in enumerate(epoch_iterator):
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            batch_loss, _ = self.loss_batch(batch['target'],
                                            batch['target'],
                                            optimizer=optimizer,
                                            step=i)
            train_loss += batch_loss
            if self.writer:
                self.writer.add_scalar('batch_loss',
                                       batch_loss,
                                       epoch * len(train_dataloader) + i + 1)
        # compute the average loss
        epoch_loss = train_loss / len(train_dataloader)

        # update scheduler
        self.scheduler.step()

        return epoch_loss

    def evaluate(self, eval_dataloader, print_report=False):
        """Evaluate the model, return average loss and accuracy.
        """
        self.model.eval()
        epoch_iterator = tqdm(eval_dataloader, desc="Iteration")

        with torch.no_grad():
            eval_loss, eval_corrects = 0., 0.
            for i, batch in enumerate(epoch_iterator):
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                batch_loss, outputs = self.loss_batch(batch['target'],
                                                      batch['target'])
                _, preds = torch.max(outputs, -1)  # preds: [batch_size, seq_len]

                if i in range(3):
                    print("- Example #{}: ".format(i+1))
                    print("- {}".format(preds[i][:batch['length'][i]].tolist()))
                    print("- {}".format(batch['target'][i][:batch['length'][i]].tolist()))

                eval_loss += batch_loss
                batch_correct = 0.
                for p, t, l in zip(preds, batch['target'], batch['length']):
                    batch_correct += torch.sum(p[:l] == t[:l]).item()
                eval_corrects += batch_correct / torch.sum(batch['length']).item()

                # eval_corrects += torch.mean((preds == batch['target']).float()).item()

            # update metrics
            avg_loss = eval_loss / len(eval_dataloader)
            avg_acc = eval_corrects / len(eval_dataloader)

        return avg_loss, avg_acc

    def fit(self, train_dataloader, eval_dataloader):
        """Model training.
        """
        best_acc = 0.
        num_epochs = self.args.num_epochs

        for epoch in range(num_epochs):
            self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

            # train
            train_loss = self.train_epoch(train_dataloader, self.optimizer, epoch)
            self.logger.info("Traing Loss: {}".format(train_loss))

            # evaluation
            eval_loss, eval_acc = self.evaluate(eval_dataloader, print_report=True)
            self.logger.info("Evaluation:")
            self.logger.info("- loss: {}".format(eval_loss))
            self.logger.info("- acc: {}".format(eval_acc))

            # monitor loss and accuracy
            if self.writer:
                self.writer.add_scalar('epoch_loss', train_loss, epoch)
                self.writer.add_scalar('eval_loss', eval_loss, epoch)
                self.writer.add_scalar('eval_acc', eval_acc, epoch)
                self.writer.add_scalar('lr', self.scheduler.get_lr()[0], epoch)

            # save the best model
            if eval_acc >= best_acc:
                best_acc = eval_acc
                self.logger.info("New best score!")
                self.save_model(self.args.save_dir)
                self.logger.info("- model is saved at: {}".format(self.args.save_dir))

    def predict(self, inputs):
        """Model inference.

        Args:
            inputs: [batch_size, seq_len]
            outputs: [batch_size, seq_len]
        """
        self.model.eval()
        with torch.no_grad():
            mask = self._build_mask(inputs, self.args.pad_idx)
            log_probs, _ = self.model(inputs, mask)  # outputs: [batch_size, seq_len, vocab_size]
            _, preds = torch.max(log_probs, -1)  # preds: [batch_size, seq_len]
        return preds
