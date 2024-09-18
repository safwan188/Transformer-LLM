from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp
class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len: int, with_residuals: bool = False, dropout_rate: float = 0.1):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len, dropout_rate)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.with_residuals = with_residuals

    def forward(self, inputs):
        if self.with_residuals:
            # Residual connection for the attention layer
            x = inputs
            attention_output = self.causal_attention(self.layer_norm_1(x))
            attention_output = self.dropout(attention_output)  # Dropout after self-attention
            x = x + attention_output

            # Residual connection for the MLP layer
            mlp_output = self.mlp(self.layer_norm_2(x))
            mlp_output = self.dropout(mlp_output)  # Dropout after MLP
            x = x + mlp_output

            return x
        else:
            x = inputs
            x = self.layer_norm_1(x)
            x = self.causal_attention(x)
            x = self.dropout(x)  # Dropout after self-attention
            x = self.layer_norm_2(x)
            x = self.mlp(x)
            x = self.dropout(x)  # Dropout after MLP
            return x

import torch
import torch.nn as nn

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x):
        b, n = x.shape
        
        # Token embeddings
        tok_embeddings = self.token_embeddings(x)
        
        # Position embeddings
        positions = torch.arange(0, n, dtype=torch.long, device=x.device)
        pos_embeddings = self.position_embeddings(positions)
        
        # Add token and position embeddings
        embeddings = tok_embeddings + pos_embeddings.unsqueeze(0)
        
        return embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            dropout_rate: float = 0.1
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.embed_dropout = nn.Dropout(dropout_rate)  # Dropout after embedding
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals, dropout_rate) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)
        x = self.embed_dropout(x)  # Apply dropout right after the embedding layer
        for layer in self.layers:
            x = layer(x)
            
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits


    def init_weights(self):
        # initialize weights
        # TODO implement initialization logic for embeddings and linear layers.
        # The code break down the parameters by type (layer-norm, linear, embedding),
        # but can also condition on individual names, for example by checking pn.endswith(...).
        for pn, p in self.named_parameters():
            if isinstance(p, nn.LayerNorm):
                torch.nn.init.zeros_(p.bias)
                torch.nn.init.ones_(p.weight)
            elif isinstance(p, nn.Linear):
                # TODO initialize p.weight and p.bias (if it is not None).
                # You can look at initializers in torch.nn.init
                pass
            elif isinstance(p, nn.Embedding):
                # TODO initialize p.weight and p.bias (if it is not None).
                # You can look at initializers in torch.nn.init
                pass


    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated
    def sample_continuation_gpu(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        device = next(self.parameters()).device  # Get the device of the model
        feed_to_lm = torch.tensor([prefix], dtype=torch.long, device=device)
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if feed_to_lm.size(1) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[:, -self.max_context_len:]
                logits = self(feed_to_lm)
                logits_for_last_token = logits[0, -1]
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token.item())
                feed_to_lm = torch.cat([feed_to_lm, sampled_token.unsqueeze(0)], dim=1)
        return generated
    def get_attention_matrices(self, sentence, tokenizer):
        # Tokenize the input sentence
        tokens = tokenizer.tokenize(sentence)
        input_ids = torch.tensor([tokens]).to(next(self.parameters()).device)

        # Store attention matrices
        attention_matrices = []

        def attention_hook(module, input, output):
            # Get the attention weights from the module
            attention_matrices.append(module.get_last_attention_weights().detach().cpu())

        # Register hooks for all attention layers
        hooks = []
        for layer in self.layers:
            hook = layer.causal_attention.register_forward_hook(attention_hook)
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            self(input_ids)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Process attention matrices
        processed_matrices = []
        for layer_idx, layer_attention in enumerate(attention_matrices):
            layer_data = {
                'layer': layer_idx,
                'attention': layer_attention.numpy(),
                'tokens': tokens
            }
            processed_matrices.append(layer_data)

        return processed_matrices

    def get_attention_matrices_batch(self, sentences, tokenizer):
        all_attention_matrices = []
        for sentence in sentences:
            attention_matrices = self.get_attention_matrices(sentence, tokenizer)
            all_attention_matrices.append({
                'sentence': sentence,
                'matrices': attention_matrices
            })
        return all_attention_matrices
    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        device = next(self.parameters()).device
        context = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
        generated = []
        
        with torch.no_grad():
            for _ in range(max_tokens_to_generate):
                if context.size(1) > self.max_context_len:
                    context = context[:, -self.max_context_len:]
                
                # Get the predictions
                logits = self(context)[:, -1, :]
                
                # Apply temperature
                logits = logits / temperature
                
                # Get top K logits and indices
                top_k_logits, top_k_indices = torch.topk(logits, topK)
                
                # Convert to probabilities
                probs = F.softmax(top_k_logits, dim=-1)
                
                # Sample from the top K
                idx_next = torch.multinomial(probs, num_samples=1).squeeze()
                
                # Convert sampled index back to vocab space
                next_token = top_k_indices[0, idx_next].item()
                
                # Append to generated sequence
                generated.append(next_token)
                
                # Update context for next iteration
                context = torch.cat((context, torch.tensor([[next_token]], device=device)), dim=1)
                
                # Break if we generate an EOS token (assuming it's the last token in the vocabulary)
                if next_token == self.word_prediction.out_features - 1:
                    break
        
        return prefix + generated
