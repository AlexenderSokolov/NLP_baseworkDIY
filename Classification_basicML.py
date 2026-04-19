import Classification_basic as cb
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import logging



import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',
    'SimHei',
]
plt.rcParams["font.family"] = [
    "DejaVu Sans",
    "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]

plt.rcParams["axes.unicode_minus"] = True


logger = logging.getLogger(__name__)


def create_criterion(loss_name):
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    if loss_name == 'cross_entropy_ls':
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    raise ValueError(f'Unsupported loss_name: {loss_name}')


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluate(model, X, y, batch_size=256):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        all_labels = []
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted.cpu())
            all_labels.append(batch_y.cpu())

        predicted_tensor = torch.cat(all_predictions)
        labels_tensor = torch.cat(all_labels)
        total = labels_tensor.size(0)
        correct = (predicted_tensor == labels_tensor).sum().item()
        accuracy = correct / total if total > 0 else 0.0
    return accuracy, predicted_tensor.numpy(), labels_tensor.numpy()

def plot_training_history(history, save_path=None, show_plot=False):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['validate_loss'], label='Validate Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['validate_accuracy'], label='Validate Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160)
    if show_plot:
        plt.show()
    plt.close()


def build_bow_matrix(X, vocab_size):
    """This function builds a Bag-of-Words matrix from the input data X, where each row corresponds to a sample and each column corresponds to a token in the vocabulary. The value at (i, j) represents the count of token j in sample i."""
    matrix = np.zeros((len(X), vocab_size), dtype=np.float32)
    for i, sample in enumerate(X):
        for token_id in sample:
            if 0 <= token_id < vocab_size:
                matrix[i, token_id] += 1.0
    return matrix


def build_ngram_vocab(X_train, n=2, min_freq=1):
    """The process is:Use a sliding window of size n to extract n-grams from each sample in the training data. Count the frequency of each n-gram across the entire training set. Filter out n-grams that occur less than min_freq times to reduce noise and limit the vocabulary size. Assign a unique ID to each remaining n-gram to create the n-gram vocabulary."""
    ngram_counts = {}
    for sample in X_train:
        if len(sample) < n:
            continue
        for i in range(len(sample) - n + 1):
            ngram = tuple(sample[i:i + n])
            # we use a tuple to represent the n-gram, which allows us to use it as a key in the dictionary. The count of each n-gram is updated in the ngram_counts dictionary.
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

    ngram_to_id = {}
    for ngram, freq in ngram_counts.items():
        if freq >= min_freq:
            ngram_to_id[ngram] = len(ngram_to_id)
    return ngram_to_id


def build_ngram_bow_matrix(X, ngram_to_id, n=2):
    """The process is:Use the n-gram vocabulary to create a BoW matrix for the input data X, where each row corresponds to a sample and each column corresponds to an n-gram in the vocabulary. The value at (i, j) represents the count of n-gram j in sample i."""
    matrix = np.zeros((len(X), len(ngram_to_id)), dtype=np.float32)
    if len(ngram_to_id) == 0:
        return matrix

    for row_idx, sample in enumerate(X):
        if len(sample) < n:
            continue
        for i in range(len(sample) - n + 1):
            ngram = tuple(sample[i:i + n])
            ngram_id = ngram_to_id.get(ngram)
            if ngram_id is not None:
                matrix[row_idx, ngram_id] += 1.0
    return matrix


def build_multi_ngram_feature_matrices(X_train, X_validate, X_test, n_list, min_freq):
    """Muti-n feature fusion: For each n in n_list, we build a separate n-gram vocabulary and corresponding BoW matrices for the training, validation, and test sets. We then concatenate these matrices along the feature dimension to create a fused feature representation that captures information from multiple n-gram levels. The function returns the fused feature matrices for the training, validation, and test sets, as well as a dictionary mapping each n to its corresponding n-gram vocabulary."""
    # Build and concatenate features for each n in n_list.
    train_parts = []
    validate_parts = []
    test_parts = []
    vocab_info = {}

    for n in n_list:
        ngram_to_id = build_ngram_vocab(X_train, n=n, min_freq=min_freq)
        if len(ngram_to_id) == 0:
            continue
        train_parts.append(build_ngram_bow_matrix(X_train, ngram_to_id, n=n))
        validate_parts.append(build_ngram_bow_matrix(X_validate, ngram_to_id, n=n))
        test_parts.append(build_ngram_bow_matrix(X_test, ngram_to_id, n=n))
        vocab_info[n] = ngram_to_id

    if not train_parts:
        raise ValueError('All N-gram vocabularies are empty. Try smaller n or lower min_freq.')

    X_train_fused = np.concatenate(train_parts, axis=1)
    X_validate_fused = np.concatenate(validate_parts, axis=1)
    X_test_fused = np.concatenate(test_parts, axis=1)
    return X_train_fused, X_validate_fused, X_test_fused, vocab_info


def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def Bag_of_Words_Classifier(
    X_train,
    y_train,
    X_validate,
    y_validate,
    X_test,
    y_test,
    vocab_size,
    num_classes,
    epochs=30,
    batch_size=64,
    learning_rate=1e-3,
    loss_name='cross_entropy',
    l2_lambda=1e-4,
    output_dir='output/figures',
    log_dir='output/logs',
    run_tag=None,
    save_plots=True,
):
    """First,we turn each sample into a fixed-length vector using the Bag-of-Words (BoW) representation, where each dimension corresponds to a token in the vocabulary and the value is the count of that token in the sample. Then, we train a simple linear classifier (a single fully connected layer) on top of these BoW vectors to perform classification.
        Second, build a linear classifier using PyTorch, and train it on the BoW representations of the training data. We will evaluate the model on the validation and test sets, and plot the training history and confusion matrices for both sets.
        input: seq_len *  dim_of_vocab ; linear layer: dim_of_vocab * num_classes; output: seq_len * num_classes
    """
    X_train_bow = build_bow_matrix(X_train, vocab_size)
    X_validate_bow = build_bow_matrix(X_validate, vocab_size)
    X_test_bow = build_bow_matrix(X_test, vocab_size)

    model = nn.Linear(vocab_size, num_classes)
    criterion = create_criterion(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # use DataLoader to create batches of data for training and evaluation, which can help improve training efficiency and stability.
    train_inputs = torch.tensor(X_train_bow, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validate_inputs = torch.tensor(X_validate_bow, dtype=torch.float32)
    validate_labels = torch.tensor(y_validate, dtype=torch.long)
    validate_dataset = TensorDataset(validate_inputs, validate_labels)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

    test_inputs = torch.tensor(X_test_bow, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    history = {
        'train_loss': [],
        'validate_loss': [],
        'train_accuracy': [],
        'validate_accuracy': [],
    }

    best_validate_acc = -1.0
    best_state = None

    os.makedirs(log_dir, exist_ok=True)
    if run_tag is None:
        run_tag = f'lr{learning_rate}_loss{loss_name}'
    log_file = os.path.join(log_dir, f'bow_{run_tag}.log')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler],
        force=True,
    )
    logger.info('Start BoW training')
    logger.info('Vocabulary size: %s, Num classes: %s, Epochs: %s, Batch size: %s, Learning rate: %s, Loss: %s, L2 lambda: %s',
                vocab_size, num_classes, epochs, batch_size, learning_rate, loss_name, l2_lambda)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_l2_loss = 0.0

        for batch_x, batch_y in train_loader:

            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # in previous process,I find evident overfitting, so I add L2 regularization to the loss function to mitigate it. 
            l2_penalty = sum(param.pow(2).sum() for param in model.parameters())
            loss = criterion(outputs, batch_y) + l2_lambda * l2_penalty
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            epoch_l2_loss += l2_penalty.item()

        train_loss = epoch_loss / train_inputs.size(0)

        model.eval()
        with torch.no_grad():
            validate_loss_total = 0.0
            validate_sample_count = 0
            for batch_x, batch_y in validate_loader:
                validate_outputs = model(batch_x)
                batch_loss = criterion(validate_outputs, batch_y)
                validate_loss_total += batch_loss.item() * batch_x.size(0)
                validate_sample_count += batch_x.size(0)
            validate_loss = validate_loss_total / max(validate_sample_count, 1)

        # we only need acc, the predicted labels and true labels can be ingnored.
        train_acc, _, _ = evaluate(model, X_train_bow, y_train, batch_size=batch_size)
        validate_acc, _, _ = evaluate(model, X_validate_bow, y_validate, batch_size=batch_size)

        history['train_loss'].append(train_loss)
        history['validate_loss'].append(validate_loss)
        history['train_accuracy'].append(train_acc)
        history['validate_accuracy'].append(validate_acc)

        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        logger.info(
            'Epoch [%s/%s] Train Loss: %.4f Val Loss: %.4f Train Acc: %.4f Val Acc: %.4f L2 Term: %.4f',
            epoch + 1,
            epochs,
            train_loss,
            validate_loss,
            train_acc,
            validate_acc,
            epoch_l2_loss,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    #still,ignore the predicted labels and true labels, we only need the accuracy for evaluation.
    train_accuracy, _, _ = evaluate(model, X_train_bow, y_train, batch_size=batch_size)
    validate_accuracy, val_pred, val_true = evaluate(model, X_validate_bow, y_validate, batch_size=batch_size)
    test_accuracy, test_pred, test_true = evaluate(model, X_test_bow, y_test, batch_size=batch_size)

    os.makedirs(output_dir, exist_ok=True)
    if save_plots:
        plot_training_history(
            history,
            save_path=os.path.join(output_dir, f'bow_{run_tag}_training_history.png'),
            show_plot=False,
        )

    class_names = [str(i) for i in range(num_classes)]
    val_cm = compute_confusion_matrix(val_true, val_pred, num_classes)
    if save_plots:
        plt.figure(figsize=(7, 6))
        plot_confusion_matrix(val_cm, classes=class_names, title='Validation Confusion Matrix')
        plt.savefig(os.path.join(output_dir, f'bow_{run_tag}_validate_confusion_matrix.png'), dpi=160)
        plt.close()

    test_cm = compute_confusion_matrix(test_true, test_pred, num_classes)
    if save_plots:
        plt.figure(figsize=(7, 6))
        plot_confusion_matrix(test_cm, classes=class_names, title='Test Confusion Matrix')
        plt.savefig(os.path.join(output_dir, f'bow_{run_tag}_test_confusion_matrix.png'), dpi=160)
        plt.close()

    #print(f"BoW Train Accuracy: {train_accuracy:.4f}")
    #print(f"BoW Validate Accuracy: {validate_accuracy:.4f}")
    #print(f"BoW Test Accuracy: {test_accuracy:.4f}")
    logger.info('Training finished')
    logger.info('BoW Train Accuracy: %.4f', train_accuracy)
    logger.info('BoW Validate Accuracy: %.4f', validate_accuracy)
    logger.info('BoW Test Accuracy: %.4f', test_accuracy)

    metrics = {
        'train_accuracy': train_accuracy,
        'validate_accuracy': validate_accuracy,
        'test_accuracy': test_accuracy,
    }
    return model, history, metrics



def N_gram_BoW_Classifier(
    X_train,
    y_train,
    X_validate,
    y_validate,
    X_test,
    y_test,
    num_classes,
    n=2, # the size of n-gram sliding window
    n_list=None,
    min_freq=1,
    epochs=30,
    batch_size=64,
    learning_rate=1e-3,
    loss_name='cross_entropy',
    l2_lambda=1e-4,
    output_dir='output/figures',
    log_dir='output/logs',
    run_tag=None,
    save_plots=True,
):
    """Train an N-gram classifier; support multi-n feature fusion via n_list."""
    # If n_list is provided, fuse multiple n-gram feature spaces.
    if n_list is None:
        n_list = [n]
    n_list = sorted(set(n_list))

    X_train_ngram, X_validate_ngram, X_test_ngram, ngram_vocab_map = build_multi_ngram_feature_matrices(
        X_train,
        X_validate,
        X_test,
        n_list=n_list,
        min_freq=min_freq,
    )
    ngram_vocab_size = X_train_ngram.shape[1]

    model = nn.Linear(ngram_vocab_size, num_classes)
    criterion = create_criterion(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_inputs = torch.tensor(X_train_ngram, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validate_inputs = torch.tensor(X_validate_ngram, dtype=torch.float32)
    validate_labels = torch.tensor(y_validate, dtype=torch.long)
    validate_dataset = TensorDataset(validate_inputs, validate_labels)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)


    # train and evaluate
    history = {
        'train_loss': [],
        'validate_loss': [],
        'train_accuracy': [],
        'validate_accuracy': [],
    }

    best_validate_acc = -1.0
    best_state = None

    os.makedirs(log_dir, exist_ok=True)
    n_tag = '_'.join(str(v) for v in n_list)
    if run_tag is None:
        run_tag = f'n{n_tag}_lr{learning_rate}_loss{loss_name}'
    log_file = os.path.join(log_dir, f'ngram_fusion_{n_tag}_training.log')
    log_file = os.path.join(log_dir, f'ngram_fusion_{run_tag}.log')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler],
        force=True,
    )
    logger.info('Start N-gram BoW training (multi-n fusion)')
    logger.info(
        'N list: %s, Min freq: %s, Fused vocab size: %s, Num classes: %s, Epochs: %s, Batch size: %s, Learning rate: %s, Loss: %s, L2 lambda: %s',
        n_list,
        min_freq,
        ngram_vocab_size,
        num_classes,
        epochs,
        batch_size,
        learning_rate,
        loss_name,
        l2_lambda,
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_l2_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            l2_penalty = sum(param.pow(2).sum() for param in model.parameters())
            loss = criterion(outputs, batch_y) + l2_lambda * l2_penalty
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            epoch_l2_loss += l2_penalty.item()

        train_loss = epoch_loss / train_inputs.size(0)

        model.eval()
        with torch.no_grad():
            validate_loss_total = 0.0
            validate_sample_count = 0
            for batch_x, batch_y in validate_loader:
                validate_outputs = model(batch_x)
                batch_loss = criterion(validate_outputs, batch_y)
                validate_loss_total += batch_loss.item() * batch_x.size(0)
                validate_sample_count += batch_x.size(0)
            validate_loss = validate_loss_total / max(validate_sample_count, 1)

        train_acc, _, _ = evaluate(model, X_train_ngram, y_train, batch_size=batch_size)
        validate_acc, _, _ = evaluate(model, X_validate_ngram, y_validate, batch_size=batch_size)

        history['train_loss'].append(train_loss)
        history['validate_loss'].append(validate_loss)
        history['train_accuracy'].append(train_acc)
        history['validate_accuracy'].append(validate_acc)

        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        logger.info(
            'Epoch [%s/%s] Train Loss: %.4f Val Loss: %.4f Train Acc: %.4f Val Acc: %.4f L2 Term: %.4f',
            epoch + 1,
            epochs,
            train_loss,
            validate_loss,
            train_acc,
            validate_acc,
            epoch_l2_loss,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_accuracy, _, _ = evaluate(model, X_train_ngram, y_train, batch_size=batch_size)
    validate_accuracy, val_pred, val_true = evaluate(model, X_validate_ngram, y_validate, batch_size=batch_size)
    test_accuracy, test_pred, test_true = evaluate(model, X_test_ngram, y_test, batch_size=batch_size)

    os.makedirs(output_dir, exist_ok=True)
    if save_plots:
        plot_training_history(
            history,
            save_path=os.path.join(output_dir, f'ngram_fusion_{run_tag}_training_history.png'),
            show_plot=False,
        )

    class_names = [str(i) for i in range(num_classes)]
    val_cm = compute_confusion_matrix(val_true, val_pred, num_classes)
    if save_plots:
        plt.figure(figsize=(7, 6))
        plot_confusion_matrix(val_cm, classes=class_names, title=f'Validation Confusion Matrix (N={n_list})')
        plt.savefig(os.path.join(output_dir, f'ngram_fusion_{run_tag}_validate_confusion_matrix.png'), dpi=160)
        plt.close()

    test_cm = compute_confusion_matrix(test_true, test_pred, num_classes)
    if save_plots:
        plt.figure(figsize=(7, 6))
        plot_confusion_matrix(test_cm, classes=class_names, title=f'Test Confusion Matrix (N={n_list})')
        plt.savefig(os.path.join(output_dir, f'ngram_fusion_{run_tag}_test_confusion_matrix.png'), dpi=160)
        plt.close()

    #print(f'N-gram (N={n}) Train Accuracy: {train_accuracy:.4f}')
    #print(f'N-gram (N={n}) Validate Accuracy: {validate_accuracy:.4f}')
    #print(f'N-gram (N={n}) Test Accuracy: {test_accuracy:.4f}')
    logger.info('N-gram training finished')
    logger.info('N-gram fusion (N=%s) Train Accuracy: %.4f', n_list, train_accuracy)
    logger.info('N-gram fusion (N=%s) Validate Accuracy: %.4f', n_list, validate_accuracy)
    logger.info('N-gram fusion (N=%s) Test Accuracy: %.4f', n_list, test_accuracy)

    metrics = {
        'train_accuracy': train_accuracy,
        'validate_accuracy': validate_accuracy,
        'test_accuracy': test_accuracy,
        'fused_vocab_size': ngram_vocab_size,
    }
    return model, history, ngram_vocab_map, metrics




def main():

    # Load and preprocess the dataset
    
    train_dataset, validate_dataset, test_dataset=cb.bulid_dataset('new_train.tsv', 'new_test.tsv', tokenize=True, output_dir='output')
    token_to_id = cb.load_from_json(os.path.join('output', 'vocab.json'))
    
    X_train, y_train = zip(*train_dataset)
    X_validate, y_validate = zip(*validate_dataset)
    X_test, y_test = zip(*test_dataset)
    
    X_train= cb.convert_samples_to_ids(X_train, token_to_id=token_to_id)
    X_validate= cb.convert_samples_to_ids(X_validate, token_to_id=token_to_id)
    X_test= cb.convert_samples_to_ids(X_test, token_to_id=token_to_id)
    
    label_set = sorted(set(y_train) | set(y_validate) | set(y_test))
    label_to_id = {label: idx for idx, label in enumerate(label_set)}

    y_train_ids = [label_to_id[label] for label in y_train]
    y_validate_ids = [label_to_id[label] for label in y_validate]
    y_test_ids = [label_to_id[label] for label in y_test]

    vocab_size = len(token_to_id)
    num_classes = len(label_set)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of classes: {num_classes}")

    learning_rate_list = [1e-3, 5e-4]
    loss_name_list = ['cross_entropy', 'cross_entropy_ls']
    result_rows = []

    for lr in learning_rate_list:
        for loss_name in loss_name_list:
            run_tag = f'lr{lr}_loss{loss_name}'

            _, _, bow_metrics = Bag_of_Words_Classifier(
                X_train,
                y_train_ids,
                X_validate,
                y_validate_ids,
                X_test,
                y_test_ids,
                vocab_size=vocab_size,
                num_classes=num_classes,
                epochs=30,
                batch_size=64,
                learning_rate=lr,
                loss_name=loss_name,
                l2_lambda=1e-4,
                output_dir=os.path.join('output', 'figures'),
                log_dir=os.path.join('output', 'logs'),
                run_tag=run_tag,
                save_plots=False,
            )
            result_rows.append({
                'feature': 'BoW',
                'n_list': '-',
                'loss': loss_name,
                'learning_rate': lr,
                'train_accuracy': bow_metrics['train_accuracy'],
                'validate_accuracy': bow_metrics['validate_accuracy'],
                'test_accuracy': bow_metrics['test_accuracy'],
            })

            _, _, _, ngram_metrics = N_gram_BoW_Classifier(
                X_train,
                y_train_ids,
                X_validate,
                y_validate_ids,
                X_test,
                y_test_ids,
                num_classes=num_classes,
                n_list=[1, 2, 3],
                min_freq=2,
                epochs=30,
                batch_size=64,
                learning_rate=lr,
                loss_name=loss_name,
                l2_lambda=1e-4,
                output_dir=os.path.join('output', 'figures'),
                log_dir=os.path.join('output', 'logs'),
                run_tag=f'n123_{run_tag}',
                save_plots=False,
            )
            result_rows.append({
                'feature': 'NgramFusion',
                'n_list': '1,2,3',
                'loss': loss_name,
                'learning_rate': lr,
                'train_accuracy': ngram_metrics['train_accuracy'],
                'validate_accuracy': ngram_metrics['validate_accuracy'],
                'test_accuracy': ngram_metrics['test_accuracy'],
            })

    results_df = pd.DataFrame(result_rows)
    results_df = results_df.sort_values(by=['validate_accuracy', 'test_accuracy'], ascending=False)
    print('\n=== Experiment Results (sorted by validate/test) ===')
    print(results_df.to_string(index=False))

    os.makedirs('output', exist_ok=True)
    results_csv_path = os.path.join('output', 'experiment_results.csv')
    results_md_path = os.path.join('output', 'experiment_results.md')
    results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
    with open(results_md_path, 'w', encoding='utf-8') as f:
        f.write(results_df.to_markdown(index=False))
    print(f'\nResults saved to: {results_csv_path}')
    print(f'Results saved to: {results_md_path}')


if __name__ == '__main__':
    main()
    

