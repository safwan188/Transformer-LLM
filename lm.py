from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def batch_to_labeled_samples(batch: torch.IntTensor) -> [torch.IntTensor, torch.IntTensor]:
    # The input batch has shape (b x n) where n is max_context + 1
    # We want to create inputs of shape (b x n-1) and labels of shape (b x n-1)
    
    inputs = batch[:, :-1]  # All columns except the last one
    labels = batch[:, 1:]   # All columns except the first one
    
    return (inputs, labels)

def compute_loss(logits, gold_labels):
    # logits size is (batch, seq_len, vocab_size)
    # gold_labels size is (batch, seq_len)
    
    # Ensure tensors are contiguous in memory
    logits = logits.contiguous()
    gold_labels = gold_labels.contiguous()
    
    # Reshape logits to (batch * seq_len, vocab_size)
    logits_flat = logits.view(-1, logits.size(-1))
    
    # Reshape gold_labels to (batch * seq_len,)
    labels_flat = gold_labels.view(-1)
    
    # Create a mask for non-padding tokens (assuming 0 is the padding token)
    mask = (labels_flat != 0)
    
    # Filter out padding tokens
    logits_flat = logits_flat[mask]
    labels_flat = labels_flat[mask]
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits_flat, labels_flat)
    
    return loss