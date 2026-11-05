"""Shared loss definitions preserved from the original notebooks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignLanguageTranslationLoss(nn.Module):
    def __init__(self, pad_token_id, label_smoothing=0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_token_id, label_smoothing=label_smoothing
        )

    def forward(self, logits, targets):
        return self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=-1):
        """
        :param vocab_size: Size of the vocabulary (number of classes)
        :param smoothing: Smoothing factor for label smoothing
        :param ignore_index: Index to ignore in the loss calculation (e.g., padding index)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        :param logits: Predicted logits of shape (batch_size, seq_len, vocab_size)
        :param target: Ground truth target indices of shape (batch_size, seq_len)
        """
        # Flatten logits and targets for easier computation
        logits = logits.view(-1, self.vocab_size)
        target = target.view(-1)

        # Mask the padding tokens
        mask = (target != self.ignore_index).unsqueeze(1)
        logits = logits[mask.expand_as(logits)].view(-1, self.vocab_size)
        target = target[mask.squeeze()]

        # Apply label smoothing
        confidence = 1.0 - self.smoothing
        smoothed_labels = torch.full_like(
            logits, self.smoothing / (self.vocab_size - 1)
        )
        smoothed_labels.scatter_(1, target.unsqueeze(1), confidence)

        # Compute KL-divergence loss
        loss = F.kl_div(
            F.log_softmax(logits, dim=-1), smoothed_labels, reduction="batchmean"
        )

        return loss


class ConsistencyLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        # Normalize features
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)

        # Compute similarity matrix
        similarity = (
            torch.matmul(features1, features2.transpose(-2, -1)) / self.temperature
        )

        # Compute consistency loss
        labels = torch.arange(similarity.size(0), device=similarity.device)
        loss = F.cross_entropy(similarity, labels)
        return loss
