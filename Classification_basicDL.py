import Classification_basic as cb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
import tqdm

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',
    'SimHei',
]
plt.rcParams["font.family"] = [
    "DejaVu Sans",
    "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]

plt.rcParams["axes.unicode_minus"] = True


def create_criterion(loss_name):
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    if loss_name == 'cross_entropy_ls':
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    raise ValueError(f'Unsupported loss_name: {loss_name}')


def create_optimizer(optimizer_name, parameters, learning_rate):
    if optimizer_name == 'adam':
        return torch.optim.Adam(parameters, lr=learning_rate)
    if optimizer_name == 'sgd':
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    if optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=learning_rate)
    raise ValueError(f'Unsupported optimizer_name: {optimizer_name}')


def pad_sequences(samples, max_len, pad_id):
    padded = np.full((len(samples), max_len), pad_id, dtype=np.int64)
    for i, sample in enumerate(samples):
        if not sample:
            continue
        # Truncate(shorten or cut off a part of sample) if longer than max_len, otherwise keep as is (with padding).
        truncated = sample[:max_len]
        padded[i, :len(truncated)] = truncated
    return padded


def convert_texts_to_ids_with_nltk_tokenizer(samples, token_to_id):
    # Explicitly use tokenizer from Classification_basic (NLTK with fallback).
    unk_id = token_to_id.get('UNK', 0)
    converted = []
    for text in samples:
        tokens = cb.safe_word_tokenize(text)
        converted.append([token_to_id.get(token, unk_id) for token in tokens])
    return converted


def get_sequence_lengths(batch_x, pad_idx):
    return (batch_x != pad_idx).sum(dim=1).clamp(min=1)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters, kernel_sizes, pad_idx, dropout=0.5):
        """
          Use Cnn for text classification.
          Args:
              vocab_size: Size of the vocabulary.
              embedding_dim: Dimension of the word embeddings.
              num_classes: Number of output classes.
              num_filters: Number of filters for each kernel size.
              kernel_sizes: List of kernel sizes (e.g., [3, 4, 5]).
              pad_idx: Index used for padding tokens.
              dropout: Dropout rate after convolution and pooling.
        """
        super().__init__()
        
        # padding_idx=pad_idx ensures that the embedding for the padding token is not updated during training and is initialized to zeros, which helps the model learn to ignore padded positions. The embedding layer maps input token IDs to dense vectors of size embedding_dim, creating a continuous representation of the input text that can capture semantic relationships between words. This is a crucial step before applying convolutional layers, as it allows the model to learn meaningful features from the text data.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # 1*1 conv * kernel_sizes, each with num_filters output channels.
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        # fc layer input size is num_filters * number of kernel sizes, output size is num_classes.
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x)          # [B, L, E] batch_size, seq_len, embedding_dim
        emb = emb.transpose(1, 2)        # [B, E, L]
        conv_features = []
        for conv in self.convs:
            c = torch.relu(conv(emb))    # [B, F, L-k+1]
            p = torch.max(c, dim=2).values
            conv_features.append(p)
        features = torch.cat(conv_features, dim=1)
        features = self.dropout(features)
        return self.fc(features)


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers, pad_idx, dropout=0.5, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # batch_first=True means input and output tensors are provided as (batch, seq, feature).
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional, # If True, becomes a bidirectional GRU. The output features will be hidden_size * 2 if bidirectional, otherwise hidden_size.
        )
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x, lengths=None):
        emb = self.embedding(x)
        if lengths is None:
            lengths = (x != self.embedding.padding_idx).sum(dim=1).clamp(min=1)
        lengths_cpu = lengths.detach().cpu()
        
        # enforce_sorted=False allows input sequences to be in any order (not necessarily sorted by length), which is more convenient for training.
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=False)
        
        #  rnn returns output and hidden state. We only need the hidden state for classification, which contains the final hidden states for each layer (and direction if bidirectional).
        _, hidden = self.rnn(packed)
        if self.bidirectional:
            
            # view the hidden state as (num_layers, num_directions, batch_size, hidden_size) and concatenate the last layer's forward and backward hidden states.
            hidden = hidden.view(self.rnn.num_layers, 2, x.size(0), self.hidden_size)
            last_hidden = torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=1)
        else:
            last_hidden = hidden[-1]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_heads, num_layers, ffn_dim, pad_idx, dropout=0.5, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.pad_idx = pad_idx
        self.max_len = max_len

    def forward(self, x, lengths=None):
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # transformer's special positional encoding is added to the word embeddings to provide the model with information about the position of each token in the sequence, which is crucial for capturing the order of words. The padding mask is created to indicate which positions are padding tokens, allowing the transformer encoder to ignore them during attention calculations. The encoded output from the transformer is then pooled by taking a weighted average of the token representations, where the weights are determined by the mask that indicates valid (non-padding) tokens. Finally, a fully connected layer maps the pooled representation to the output classes for classification.
        emb = self.embedding(x) + self.position_embedding(positions.clamp(max=self.max_len - 1))
        
        # eq() creates a boolean mask indicating which positions are padding tokens
        padding_mask = x.eq(self.pad_idx)
        encoded = self.encoder(emb, src_key_padding_mask=padding_mask)
        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


def build_model(config, vocab_size, pad_idx, num_classes, device, max_len):
    model_name = config['model_name']
    if model_name == 'cnn':
        model = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            num_classes=num_classes,
            num_filters=config['num_filters'],
            kernel_sizes=config['kernel_sizes'],
            pad_idx=pad_idx,
            dropout=config['dropout'],
        )
    elif model_name == 'rnn':
        model = TextRNN(
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_size=config['hidden_size'],
            num_classes=num_classes,
            num_layers=config['num_layers'],
            pad_idx=pad_idx,
            dropout=config['dropout'],
            bidirectional=config.get('bidirectional', True),
        )
    elif model_name == 'transformer':
        if config['embedding_dim'] % config['num_heads'] != 0:
            raise ValueError(
                f"embedding_dim ({config['embedding_dim']}) must be divisible by num_heads ({config['num_heads']})."
            )
        model = TextTransformer(
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            num_classes=num_classes,
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            ffn_dim=config['ffn_dim'],
            pad_idx=pad_idx,
            dropout=config['dropout'],
            max_len=max_len,
        )
    else:
        raise ValueError(f'Unsupported model_name: {model_name}')
    return model.to(device)


def forward_model(model, batch_x, config, pad_idx):
    model_name = config['model_name']
    if model_name == 'rnn':
        lengths = get_sequence_lengths(batch_x, pad_idx)
        return model(batch_x, lengths)
    return model(batch_x)


def evaluate(model, data_loader, criterion, device):
    raise RuntimeError('Use evaluate_model with model config.')


def evaluate_model(model, data_loader, criterion, device, config, pad_idx):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
          # non_blocking=True allows faster data transfer to GPU if the data loader is pinned in memory, 
          # but it requires careful handling to avoid potential issues.
          # Here we ensure that the data loader is created with pin_memory=True when using CUDA, 
          # and we use non_blocking=True in to() calls for better performance. 
          # This is a common practice to speed up training when using GPUs.
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            logits = forward_model(model, batch_x, config, pad_idx)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_count += batch_x.size(0)
    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def train_one_experiment(
    train_loader,
    validate_loader,
    test_loader,
    vocab_size,
    pad_idx,
    num_classes,
    config,
    device,
):
    model = build_model(config, vocab_size, pad_idx, num_classes, device, config['max_len'])

    criterion = create_criterion(config['loss_name'])
    optimizer = create_optimizer(config['optimizer_name'], model.parameters(), config['learning_rate'])

    best_val_acc = -1.0
    best_state = None

    epoch_iterator = tqdm.tqdm(range(config['epochs']), desc='Epochs', leave=False)
    for _ in epoch_iterator:
        model.train()
        train_iterator = tqdm.tqdm(train_loader, desc='Train Batches', leave=False)
        for batch_x, batch_y in train_iterator:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = forward_model(model, batch_x, config, pad_idx)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        _, val_acc = evaluate_model(model, validate_loader, criterion, device, config, pad_idx)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_loss, train_acc = evaluate_model(model, train_loader, criterion, device, config, pad_idx)
    val_loss, val_acc = evaluate_model(model, validate_loader, criterion, device, config, pad_idx)
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, config, pad_idx)

    return {
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'validate_loss': val_loss,
        'validate_accuracy': val_acc,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
    }


def main():
    os.makedirs('output', exist_ok=True)
    os.makedirs(os.path.join('output', 'logs'), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join('output', 'logs', 'dl_cnn_experiments.log'), mode='w', encoding='utf-8')],
        force=True,
    )
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: %s', device)

    # Read text first, then explicitly tokenize with cb.safe_word_tokenize.
    train_dataset, validate_dataset, test_dataset = cb.bulid_dataset(
        'new_train.tsv', 'new_test.tsv', tokenize=False, output_dir='output'
    )
    token_to_id = cb.load_from_json(os.path.join('output', 'vocab.json'))

    X_train, y_train = zip(*train_dataset)
    X_validate, y_validate = zip(*validate_dataset)
    X_test, y_test = zip(*test_dataset)

    X_train_ids = convert_texts_to_ids_with_nltk_tokenizer(X_train, token_to_id=token_to_id)
    X_validate_ids = convert_texts_to_ids_with_nltk_tokenizer(X_validate, token_to_id=token_to_id)
    X_test_ids = convert_texts_to_ids_with_nltk_tokenizer(X_test, token_to_id=token_to_id)

    logger.info('Tokenizer in use: Classification_basic.safe_word_tokenize (NLTK-based)')

    label_set = sorted(set(y_train) | set(y_validate) | set(y_test))
    label_to_id = {label: idx for idx, label in enumerate(label_set)}
    y_train_ids = np.array([label_to_id[label] for label in y_train], dtype=np.int64)
    y_validate_ids = np.array([label_to_id[label] for label in y_validate], dtype=np.int64)
    y_test_ids = np.array([label_to_id[label] for label in y_test], dtype=np.int64)

    pad_idx = token_to_id.get('PAD', len(token_to_id) - 1)
    vocab_size = len(token_to_id)
    num_classes = len(label_set)

    # Use percentile length to balance speed and coverage.
    train_lengths = [len(x) for x in X_train_ids]
    max_len = int(np.percentile(train_lengths, 95))
    max_len = max(max_len, 8)

    X_train_padded = pad_sequences(X_train_ids, max_len=max_len, pad_id=pad_idx)
    X_validate_padded = pad_sequences(X_validate_ids, max_len=max_len, pad_id=pad_idx)
    X_test_padded = pad_sequences(X_test_ids, max_len=max_len, pad_id=pad_idx)

    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_padded, dtype=torch.long), torch.tensor(y_train_ids, dtype=torch.long)),
        batch_size=128,
        shuffle=True,
        pin_memory=use_cuda,
    )
    validate_loader = DataLoader(
        TensorDataset(torch.tensor(X_validate_padded, dtype=torch.long), torch.tensor(y_validate_ids, dtype=torch.long)),
        batch_size=256,
        shuffle=False,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_padded, dtype=torch.long), torch.tensor(y_test_ids, dtype=torch.long)),
        batch_size=256,
        shuffle=False,
        pin_memory=use_cuda,
    )

    loss_name_list = ['cross_entropy', 'cross_entropy_ls']
    learning_rate_list = [1e-3, 5e-4]
    optimizer_name_list = ['adam', 'rmsprop']

    cnn_num_filters_list = [64, 128]
    cnn_kernel_sizes_list = [[3, 4, 5], [2, 3, 4]]

    rnn_hidden_size_list = [128, 256]
    rnn_num_layers_list = [1, 2]

    transformer_num_heads_list = [4, 8]
    transformer_num_layers_list = [1, 2]
    transformer_ffn_dim_list = [256, 512]

    result_rows = []
    total_runs = (
        len(loss_name_list)
        * len(learning_rate_list)
        * len(optimizer_name_list)
        * (len(cnn_num_filters_list) * len(cnn_kernel_sizes_list)
           + len(rnn_hidden_size_list) * len(rnn_num_layers_list)
           + len(transformer_num_heads_list) * len(transformer_num_layers_list) * len(transformer_ffn_dim_list))
    )
    run_iterator = tqdm.tqdm(total=total_runs, desc='Experiment Runs')

    for loss_name in loss_name_list:
        for learning_rate in learning_rate_list:
            for optimizer_name in optimizer_name_list:
                for num_filters in cnn_num_filters_list:
                    for kernel_sizes in cnn_kernel_sizes_list:
                        config = {
                            'model_name': 'cnn',
                            'loss_name': loss_name,
                            'learning_rate': learning_rate,
                            'optimizer_name': optimizer_name,
                            'num_filters': num_filters,
                            'kernel_sizes': kernel_sizes,
                            'embedding_dim': 128,
                            'dropout': 0.5,
                            'epochs': 8,
                            'max_len': max_len,
                        }
                        logger.info('Running config: %s', config)
                        metrics = train_one_experiment(
                            train_loader=train_loader,
                            validate_loader=validate_loader,
                            test_loader=test_loader,
                            vocab_size=vocab_size,
                            pad_idx=pad_idx,
                            num_classes=num_classes,
                            config=config,
                            device=device,
                        )
                        result_rows.append({
                            'model': 'cnn',
                            'loss': loss_name,
                            'learning_rate': learning_rate,
                            'optimizer': optimizer_name,
                            'num_filters': num_filters,
                            'kernel_sizes': str(kernel_sizes),
                            'hidden_size': '-',
                            'num_layers': '-',
                            'num_heads': '-',
                            'ffn_dim': '-',
                            'train_loss': metrics['train_loss'],
                            'train_accuracy': metrics['train_accuracy'],
                            'validate_loss': metrics['validate_loss'],
                            'validate_accuracy': metrics['validate_accuracy'],
                            'test_loss': metrics['test_loss'],
                            'test_accuracy': metrics['test_accuracy'],
                        })
                        run_iterator.update(1)

                for hidden_size in rnn_hidden_size_list:
                    for num_layers in rnn_num_layers_list:
                        config = {
                            'model_name': 'rnn',
                            'loss_name': loss_name,
                            'learning_rate': learning_rate,
                            'optimizer_name': optimizer_name,
                            'hidden_size': hidden_size,
                            'num_layers': num_layers,
                            'bidirectional': True,
                            'embedding_dim': 128,
                            'dropout': 0.5,
                            'epochs': 8,
                            'max_len': max_len,
                        }
                        logger.info('Running config: %s', config)
                        metrics = train_one_experiment(
                            train_loader=train_loader,
                            validate_loader=validate_loader,
                            test_loader=test_loader,
                            vocab_size=vocab_size,
                            pad_idx=pad_idx,
                            num_classes=num_classes,
                            config=config,
                            device=device,
                        )
                        result_rows.append({
                            'model': 'rnn',
                            'loss': loss_name,
                            'learning_rate': learning_rate,
                            'optimizer': optimizer_name,
                            'num_filters': '-',
                            'kernel_sizes': '-',
                            'hidden_size': hidden_size,
                            'num_layers': num_layers,
                            'num_heads': '-',
                            'ffn_dim': '-',
                            'train_loss': metrics['train_loss'],
                            'train_accuracy': metrics['train_accuracy'],
                            'validate_loss': metrics['validate_loss'],
                            'validate_accuracy': metrics['validate_accuracy'],
                            'test_loss': metrics['test_loss'],
                            'test_accuracy': metrics['test_accuracy'],
                        })
                        run_iterator.update(1)

                for num_heads in transformer_num_heads_list:
                    for num_layers in transformer_num_layers_list:
                        for ffn_dim in transformer_ffn_dim_list:
                            config = {
                                'model_name': 'transformer',
                                'loss_name': loss_name,
                                'learning_rate': learning_rate,
                                'optimizer_name': optimizer_name,
                                'num_heads': num_heads,
                                'num_layers': num_layers,
                                'ffn_dim': ffn_dim,
                                'embedding_dim': 128,
                                'dropout': 0.5,
                                'epochs': 8,
                                'max_len': max_len,
                            }
                            logger.info('Running config: %s', config)
                            metrics = train_one_experiment(
                                train_loader=train_loader,
                                validate_loader=validate_loader,
                                test_loader=test_loader,
                                vocab_size=vocab_size,
                                pad_idx=pad_idx,
                                num_classes=num_classes,
                                config=config,
                                device=device,
                            )
                            result_rows.append({
                                'model': 'transformer',
                                'loss': loss_name,
                                'learning_rate': learning_rate,
                                'optimizer': optimizer_name,
                                'num_filters': '-',
                                'kernel_sizes': '-',
                                'hidden_size': '-',
                                'num_layers': num_layers,
                                'num_heads': num_heads,
                                'ffn_dim': ffn_dim,
                                'train_loss': metrics['train_loss'],
                                'train_accuracy': metrics['train_accuracy'],
                                'validate_loss': metrics['validate_loss'],
                                'validate_accuracy': metrics['validate_accuracy'],
                                'test_loss': metrics['test_loss'],
                                'test_accuracy': metrics['test_accuracy'],
                            })
                            run_iterator.update(1)

    run_iterator.close()

    results_df = pd.DataFrame(result_rows)
    results_df = results_df.sort_values(by=['validate_accuracy', 'test_accuracy'], ascending=False)

    print('=== CNN / RNN / Transformer Experiment Results (sorted by validation/test accuracy) ===')
    print(results_df.to_string(index=False))

    csv_path = os.path.join('output', 'dl_model_experiment_results.csv')
    md_path = os.path.join('output', 'dl_model_experiment_results.md')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(results_df.to_markdown(index=False))

    logger.info('Saved result table to %s', csv_path)
    logger.info('Saved result table to %s', md_path)


if __name__ == '__main__':
    main()
  

  
  
