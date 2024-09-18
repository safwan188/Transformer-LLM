from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp
import optuna
class TransformerLMWrapper:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

    def to_device(self, data):
        return {k: v.to(self.device) for k, v in data.items()} if isinstance(data, dict) else data.to(self.device)
    
    def train_step(self, batch, optimizer):
        batch = self.to_device(batch)
        
        logits = self.model(batch['input_ids'])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch['labels'].view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item()

    def eval_step(self, batch):
        batch = self.to_device(batch)
        
        with torch.no_grad():
            logits = self.model(batch['input_ids'])
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch['labels'].view(-1))

        return loss.item()

    def generate(self, prefix, max_tokens):
        if isinstance(prefix, str):
            prefix = self.tokenizer.tokenize(prefix)  # Assuming tokenizer is available
        elif isinstance(prefix, torch.Tensor):
            prefix = prefix.cpu().tolist()
        return self.model.better_sample_continuation(prefix, max_tokens,1,5)
        return self.model.sample_continuation_gpu(prefix, max_tokens)

    def sample_continuation_gpu(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        return self.model.sample_continuation_gpu(prefix, max_tokens_to_generate)