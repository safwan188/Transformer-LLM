from __future__ import annotations
import torch
import my_model
import data
import lm
import time
from transformer import TransformerLM
from torch import optim
import itertools
import json
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import nltk
from collections import Counter
import spacy

# Download necessary NLTK data
nltk.download('words')
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
# Load Hebrew spaCy model


def score_english_similarity(text):
    english_words = set(nltk.corpus.words.words())
    tokens = nltk.word_tokenize(text.lower())
    word_score = sum(1 for word in tokens if word in english_words) / len(tokens) if tokens else 0
    doc = nlp(text)
    sentence_score = sum(1 for sent in doc.sents if len(sent) > 2) / len(list(doc.sents))
    UNIVERSAL_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    pos_diversity = len(Counter(token.pos_ for token in doc)) / len(UNIVERSAL_TAGS)
    return (word_score * 0.5 + sentence_score * 0.3 + pos_diversity * 0.2) * 100

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return max(0.001, float(current_step+1) / float(max(1, num_warmup_steps)))
        return max(
            0.0005, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
def reverse_text(text):
    return text[::-1]

def train_model(params, model_id):
    # Fixed hyperparameters
    seq_len = 128
    batch_size = 64
    data_path = "heb-data/"
    num_batches_to_train = 20000  # Reduced for quicker trials
    gradient_clipping = 1.0
    dropout_rate = 0.1
    warmup_steps = 500  # Number of warm-up steps

    n_layers, n_heads = params['config']
    embed_size = params['embed_size']
    mlp_hidden_size_factor = params['mlp_hidden_size_factor']
    learning_rate = params['learning_rate']
    optimizer_name = params['optimizer']
    weight_decay = params['weight_decay']
    patience = params['patience']
    min_delta = params['min_delta']

    tokenizer, tokenized_data = data.load_data(data_path)
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        embed_size * mlp_hidden_size_factor,
        with_residuals=True,
        dropout_rate=dropout_rate
    )

    wrapper = my_model.TransformerLMWrapper(model)

    # Choose optimizer
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(wrapper.model.parameters(), lr=learning_rate, betas=[0.9, 0.95], weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(wrapper.model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay)

    # Define the warm-up scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_batches_to_train)

    start = time.time()
    num_batches = 0 
    total_loss = 0
    losses = []
    english_scores = []
    
    best_loss = float('inf')
    no_improve_count = 0
    
    while num_batches < num_batches_to_train:
        epoch_loss = 0
        epoch_batches = 0
        
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train:
                break
            
            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_data = {'input_ids': batch_x, 'labels': batch_y}

            loss = wrapper.train_step(batch_data, optimizer)
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), gradient_clipping)
            
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss
            epoch_loss += loss
            epoch_batches += 1

            # Step the scheduler
            scheduler.step()

            num_batches += 1
            if num_batches % 100 == 0:
                losses.append(loss)
                print(f"Batches: {num_batches}, Avg Loss: {total_loss / num_batches:.4f}, Time: {time.time() - start:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
            if num_batches % 1000 == 0:
                print("Sample:")
                sample = tokenizer.detokenize(wrapper.generate(tokenizer.tokenize("Hello"), 500))
                sample_rtl = reverse_text(sample)  # Add RTL marker
                print(sample_rtl)
                # Calculate and store English score
                if data_path == "data:":
                    english_score = score_english_similarity(sample)
                    english_scores.append(english_score)
                    print(f"English score: {english_score:.2f}")
        
        # Check for early stopping after each epoch
        avg_epoch_loss = epoch_loss / epoch_batches
        if avg_epoch_loss < best_loss - min_delta:
            best_loss = avg_epoch_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            print(f"Early stopping triggered after {num_batches} batches")
            break

    # Generate sample
    wrapper.model.eval()
    sample = tokenizer.detokenize(wrapper.generate(tokenizer.tokenize("למה"), 500))

    # Prepare model data
    model_data = {
        'model_id': model_id,
        'params': params,
        'losses': losses,
        'english_scores': english_scores,  # Add English scores to model data
        'avg_loss': total_loss / num_batches,
        'training_time': time.time() - start,
        'sample': sample
    }

    # Save model data
    with open(f'model_results/model_{model_id}_data.json', 'w') as f:
        json.dump(model_data, f, indent=2)
    
    # Save the trained model
    model_save_path = f'model_results/model_{model_id}.pth'
    torch.save(wrapper.model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save .info file
    info_content = f"""Model ID: {model_id}
Config: {params['config']}
Embed Size: {params['embed_size']}
MLP Hidden Size Factor: {params['mlp_hidden_size_factor']}
Learning Rate: {params['learning_rate']}
Optimizer: {params['optimizer']}
Weight Decay: {params['weight_decay']}
Patience: {params['patience']}
Min Delta: {params['min_delta']}
Average Loss: {total_loss / num_batches:.4f}
Final English Score: {english_scores[-1]:.2f}
Training Time: {time.time() - start:.2f}s
"""
    with open(f'model_results/model_{model_id}.info', 'w') as f:
        f.write(info_content)

    return total_loss / num_batches, english_scores[-1]  # Return both avg loss and final English score

def main():
    # Create a directory to store results if it doesn't exist
    os.makedirs('model_results', exist_ok=True)

    # Define parameter grid
    param_grid = {
        'config': [(6, 6), ],
        'embed_size': [192,],
        'mlp_hidden_size_factor': [4],
        'learning_rate': [1e-3],
        'optimizer': ['AdamW'],
        'weight_decay': [0.001, ],
        'patience': [50],
        'min_delta': [0.001,]
    }

    # Generate all combinations of parameters
    param_combinations = list(itertools.product(*param_grid.values()))

    best_loss = float('inf')
    best_english_score = 0
    best_params = None

    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"\nTraining model {i+1}/{len(param_combinations)}:")
        print(param_dict)
        
        avg_loss, final_english_score = train_model(param_dict, i)
        
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Final English score: {final_english_score:.2f}")
        
        # You can adjust this condition based on how you want to balance loss vs English score
        if avg_loss < best_loss and final_english_score > best_english_score:
            best_loss = avg_loss
            best_english_score = final_english_score
            best_params = param_dict

    print("\nBest configuration:")
    print(f"  Loss: {best_loss:.4f}")
    print(f"  English score: {best_english_score:.2f}")
    print("  Params:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    main()