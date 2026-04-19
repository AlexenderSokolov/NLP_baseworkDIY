import json
from collections import Counter
import re
import os
import nltk
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',  
    'SimHei',           
]
plt.rcParams["font.family"] = [
    "DejaVu Sans",
    "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]

plt.rcParams["axes.unicode_minus"] = True



def safe_word_tokenize(text):
    """Tokenize text with NLTK and gracefully fallback when punkt resources are unavailable."""
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        return nltk.tokenize.wordpunct_tokenize(text)

class TextDataset_tsv:
  
    def __init__(self, file_path, tokenize=True):
        self.data = []
        self.labels = []
        self.read_tsv(file_path, tokenize=tokenize)

    def _clean_text(self, text):
        text = text.lower()
        text = text.replace('-lrb-', ' ')
        text = text.replace('-rrb-', ' ')
        text = text.replace('``', ' ')
        text = text.replace("''", ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _is_low_info_token(self, token):
        if not token:
            return True
        if token in {'-lrb-', '-rrb-', '``', "''"}:
            return True
        return all(not ch.isalnum() for ch in token)

    def _ensure_tokens(self, text):
        if isinstance(text, list):
            tokens = text
        else:
            tokens = safe_word_tokenize(text)
        return [token for token in tokens if not self._is_low_info_token(token)]

    def read_tsv(self, file_path, tokenize=True):
          """seperate the text and labels from the tsv file, and store them in self.data and self.labels respectively.
            What's more,when tokenize is True, the text will be tokenized and converted to lowercase, otherwise it will be stored as it is.
          """
               
          with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
              line = line.strip()
              if line:
                parts = line.split('\t', 1)
                if len(parts) == 2: 
                  text = self._clean_text(parts[0])
                  label = parts[1].strip()
                  if not text or not label:
                      continue
                  if tokenize:
                      tokens = safe_word_tokenize(text)
                      tokens = [token for token in tokens if not self._is_low_info_token(token)]
                      if not tokens:
                          continue
                      self.data.append(tokens)
                  else:
                      self.data.append(text)
                  self.labels.append(label)
    
                        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def build_vocab(self,min_freq=1):
      """Build a vocabulary from the dataset."""
      frequency = Counter()

      for text in self.data:
                frequency.update(self._ensure_tokens(text))

      self.frequency = dict(frequency)

      vocab = [word for word, freq in frequency.items() if freq >= min_freq]
      vocab.sort(key=lambda word: (-frequency[word], word))

      self.token_to_id = {word: idx for idx, word in enumerate(vocab)}
      self.token_to_id['UNK'] = len(self.token_to_id)
      self.token_to_id['PAD'] = len(self.token_to_id)

      self.id_to_token = {idx: word for word, idx in self.token_to_id.items()}  
    
    def get_word_id(self, word):
        """Get the ID of a word, return the ID of 'UNK' if the word is not in the vocabulary."""
        return self.token_to_id.get(word, self.token_to_id['UNK'])
    
    def get_id_word(self, idx):
        """Get the word corresponding to an ID, return 'UNK' if the ID is not in the vocabulary."""
        return self.id_to_token.get(idx, 'UNK')
    
    def get_word_freq(self, word):
        """Get the frequency of a word in the dataset."""
        return self.frequency.get(word, 0)
    
    def get_word_distribution(self):
        """Get the distribution of words in the dataset."""
        total_words = sum(self.frequency.values())
        if total_words == 0:
            return {}
        return {word: freq / total_words for word, freq in self.frequency.items()}
    
    def convert_tokens_to_ids(self, tokens):
        """Convert a list of tokens to a list of IDs."""
        return [self.get_word_id(token) for token in tokens]

def save_to_json(data, file_path):
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def bulid_dataset(train_validate_path, test_path, tokenize=True, output_dir='.'):
    """build the dataset from the tsv files, and save the vocabulary and the word distribution to json files."""
    train_validate_dataset = TextDataset_tsv(train_validate_path, tokenize=tokenize)
    test_dataset = TextDataset_tsv(test_path, tokenize=tokenize)
    
    # build the vocabulary from the training and validation dataset
    train_validate_dataset.build_vocab()
    
    # save the vocabulary and the word distribution to json files
    save_to_json(train_validate_dataset.token_to_id, os.path.join(output_dir, 'vocab.json'))
    save_to_json(train_validate_dataset.get_word_distribution(), os.path.join(output_dir, 'word_distribution.json'))
    
    # split the training and validation dataset into training and validation sets
    label_counts = Counter(train_validate_dataset.labels)
    stratify_labels = train_validate_dataset.labels if label_counts and min(label_counts.values()) >= 2 else None
    train_dataset, validate_dataset = train_test_split(
        train_validate_dataset,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=stratify_labels,
    )

    return train_dataset, validate_dataset, test_dataset







class NaiveBayesClassifier:
    """Naive Bayes classifier that counts the occurrences of words in each class and predicts the class with the highest likelihood."""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.class_priors = {}
        self.word_likelihoods = {}

    def fit(self, X, y):
        """Fit the Naive Bayes classifier to the training data."""
        class_counts = Counter(y)
        total_samples = len(y)

        for cls in class_counts:
            self.class_priors[cls] = class_counts[cls] / total_samples
            self.word_likelihoods[cls] = np.zeros(self.vocab_size)

        for text, label in zip(X, y):
            for word_id in text:
                self.word_likelihoods[label][word_id] += 1

        for cls in self.word_likelihoods:
            self.word_likelihoods[cls] += 1  # Laplace smoothing
            self.word_likelihoods[cls] /= (class_counts[cls] + self.vocab_size)

    def predict(self, X):
        """Predict the class labels for the given input data."""
        predictions = []
        for text in X:
            class_scores = {}
            for cls in self.class_priors:
                log_prior = np.log(self.class_priors[cls])
                log_likelihood = sum(np.log(self.word_likelihoods[cls][word_id]) for word_id in text)
                class_scores[cls] = log_prior + log_likelihood
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return predictions

class LogisticRegressionClassifier:
    """Multinomial logistic regression classifier based on sparse bag-of-words features."""
    def __init__(self, draw_pictures=False, l2=0.0, random_state=42, plot_dir='output/figures', show_plots=False) -> None:
        self.weights = None
        self.b = None
        self.draw_pictures = draw_pictures
        self.l2 = l2
        self.random_state = random_state
        self.plot_dir = plot_dir
        self.show_plots = show_plots
        self.loss_history = []
        self.classes_ = []
        self.class_to_index = {}
        self.index_to_class = {}
        self.class_weights = {}

    def _save_plot(self, filename):
        os.makedirs(self.plot_dir, exist_ok=True)
        save_path = os.path.join(self.plot_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        if self.show_plots:
            plt.show()
        plt.close()

    def _softmax(self, logits):
        shifted_logits = logits - np.max(logits)
        exp_logits = np.exp(np.clip(shifted_logits, -35, 35))
        return exp_logits / np.sum(exp_logits)

    def _doc_to_sparse_counts(self, text):
        """convert a document represented as a list of token IDs into sparse feature counts."""
        token_counts = Counter(text)
        if not token_counts:
            return np.array([], dtype=int), np.array([], dtype=float)
        
        # tips:np.fromiter is used to create arrays from iterators, which is more memory efficient than creating a list first and then converting it to an array.
        feature_ids = np.fromiter(token_counts.keys(), dtype=int)
        feature_values = np.fromiter(token_counts.values(), dtype=float)
        return feature_ids, feature_values

    def _prepare_class_weights(self, y, class_weight):
        """Initialize class weights based on the specified class_weight parameter."""
        if class_weight is None:
            self.class_weights = {label: 1.0 for label in self.classes_}
            return

        if class_weight == 'balanced':
            counts = Counter(y)
            total = len(y)
            num_classes = len(counts)
            self.class_weights = {
                label: total / (num_classes * counts[label])
                for label in counts
            }
            return
        
        if isinstance(class_weight, dict):
            self.class_weights = {
                label: float(class_weight.get(label, 1.0))
                for label in self.classes_
            }
            return

        raise ValueError("class_weight must be None, 'balanced', or a dict.")

    def _compute_average_loss(self, X, y):
        """compute the average loss over the given dataset, including the L2 regularization term."""
        eps = 1e-12
        total_loss = 0.0
        for text, label in zip(X, y):
            feature_ids, feature_values = self._doc_to_sparse_counts(text)
            logits = np.dot(self.weights[:, feature_ids], feature_values) + self.b
            probabilities = self._softmax(logits)
            target_index = self.class_to_index[label]
            sample_weight = self.class_weights.get(label, 1.0)
            total_loss += -sample_weight * np.log(probabilities[target_index] + eps)

        total_loss /= max(len(X), 1)
        # Add L2 regularization term to the loss
        total_loss += 0.5 * self.l2 * np.sum(self.weights * self.weights)
        return total_loss

    def fit(
        self,
        X,
        y,
        learning_rate=0.05,
        epochs=200,
        batch_size=32,
        lr_decay=0.0, # learning rate decay factor, which reduces the learning rate over epochs to help convergence.
        class_weight='balanced',
        X_val=None,
        y_val=None,
        early_stopping_rounds=10,
        tol=1e-4,
    ):
        """Fit with mini-batch SGD, optional class weighting, and optional early stopping."""
        
        if not X or not y:
            raise ValueError("Training data cannot be empty.")
        
        self.classes_ = sorted(set(y))
        if len(self.classes_) < 2:
            raise ValueError("LogisticRegressionClassifier requires at least two classes.")
        self.class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        self.index_to_class = {idx: label for label, idx in self.class_to_index.items()}
        self._prepare_class_weights(y, class_weight)

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        num_features = max((max(text) if text else -1) for text in X) + 1
        if num_features <= 0:
            raise ValueError("No valid features found in training data.")
        num_classes = len(self.classes_)
        self.weights = np.zeros((num_classes, num_features))
        self.b = np.zeros(num_classes)
        self.loss_history = []
        
        #use numpy's random generator for shuffling indices to ensure reproducibility and better performance compared to Python's built-in random module.
        rng = np.random.default_rng(self.random_state)
        indices = np.arange(len(X))
        sparse_docs = [self._doc_to_sparse_counts(text) for text in X]

        best_metric = float('inf')
        best_weights = self.weights.copy()
        best_b = self.b
        no_improve_rounds = 0

        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must both be provided for validation.")
        use_validation = X_val is not None and y_val is not None

        for epoch in range(epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            # the more epochs we train, the smaller the learning rate will be, which can help the model converge more smoothly and avoid overshooting minima in the loss landscape.
            current_lr = learning_rate / (1.0 + lr_decay * epoch)

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                grad_w = np.zeros_like(self.weights)
                grad_b = np.zeros_like(self.b)
                batch_loss = 0.0

                for idx in batch_indices:
                    feature_ids, feature_values = sparse_docs[idx]
                    label = y[idx]
                    sample_weight = self.class_weights.get(label, 1.0)
                    target_index = self.class_to_index[label]
                    logits = np.dot(self.weights[:, feature_ids], feature_values) + self.b
                    probabilities = self._softmax(logits)
                    errors = probabilities.copy()
                    errors[target_index] -= 1.0
                    errors *= sample_weight

                    if feature_ids.size > 0:
                        grad_w[:, feature_ids] += errors[:, None] * feature_values
                    grad_b += errors
                    batch_loss += -sample_weight * np.log(probabilities[target_index] + 1e-12)

                actual_batch_size = len(batch_indices)
                if actual_batch_size == 0:
                    continue

                grad_w = grad_w / actual_batch_size
                grad_w += self.l2 * self.weights
                grad_b = grad_b / actual_batch_size

                self.weights -= current_lr * grad_w
                self.b -= current_lr * grad_b

                epoch_loss += batch_loss

            average_epoch_loss = epoch_loss / len(X)
            regularized_epoch_loss = average_epoch_loss + 0.5 * self.l2 * np.sum(self.weights * self.weights)
            self.loss_history.append(regularized_epoch_loss)

            metric = self._compute_average_loss(X_val, y_val) if use_validation else regularized_epoch_loss
            if metric + tol < best_metric:
                best_metric = metric
                best_weights = self.weights.copy()
                best_b = self.b
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                if early_stopping_rounds is not None and no_improve_rounds >= early_stopping_rounds:
                    self.weights = best_weights
                    self.b = best_b
                    break

        if self.draw_pictures:
            plt.figure(figsize=(8, 4.5))
            plt.plot(range(1, len(self.loss_history) + 1), self.loss_history)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            self._save_plot('logistic_regression_training_loss.png')

    def predict_proba(self, X):
        """Predict class probabilities for each sample."""
        if self.weights is None or self.b is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        probabilities = []
        for text in X:
            feature_ids, feature_values = self._doc_to_sparse_counts(text)
            valid_mask = feature_ids < self.weights.shape[1]
            feature_ids = feature_ids[valid_mask]
            feature_values = feature_values[valid_mask]
            logits = np.dot(self.weights[:, feature_ids], feature_values) + self.b
            probabilities.append(self._softmax(logits))
        return probabilities

    def predict(self, X):
        """Predict the class labels for the given input data."""
        probabilities = self.predict_proba(X)
        predictions = [self.index_to_class[int(np.argmax(prob))] for prob in probabilities]

        if self.draw_pictures:
            counts = Counter(predictions)
            plt.figure(figsize=(8, 4.5))
            plt.bar(
                self.classes_,
                [counts.get(label, 0) for label in self.classes_]
            )
            plt.xlabel('Predicted Class')
            plt.ylabel('Frequency')
            plt.title('Predicted Class Distribution')
            self._save_plot('logistic_regression_predicted_distribution.png')

        return predictions


def evaluate_classifier(classifier, X, y):
    """Evaluate the classifier on the given data and return the accuracy."""
    predictions = classifier.predict(X)
    correct = sum(pred == true for pred, true in zip(predictions, y))
    return correct / len(y) if y else 0.0


def convert_samples_to_ids(samples, token_to_id):
    """Convert tokenized samples to id sequences with UNK fallback."""
    unk_id = token_to_id.get('UNK', 0)
    converted = []
    for sample in samples:
        if isinstance(sample, str):
            tokens = safe_word_tokenize(sample)
        else:
            tokens = sample
        converted.append([token_to_id.get(token, unk_id) for token in tokens])
    return converted



def main():
      train_dataset, validate_dataset, test_dataset = bulid_dataset('new_train.tsv', 'new_test.tsv', tokenize=True, output_dir='output')
      token_to_id = load_from_json(os.path.join('output', 'vocab.json'))
      print(f'Training samples: {len(train_dataset)}, Validation samples: {len(validate_dataset)}, Test samples: {len(test_dataset)}')
      print(f'Vocabulary size: {len(token_to_id)}\n')
      
      X_train, y_train = zip(*train_dataset)
      X_validate, y_validate = zip(*validate_dataset)
      X_test, y_test = zip(*test_dataset)

      X_train = convert_samples_to_ids(X_train, token_to_id)
      X_validate = convert_samples_to_ids(X_validate, token_to_id)
      X_test = convert_samples_to_ids(X_test, token_to_id)

      vocab_size = len(token_to_id)
      
      nb_classifier = NaiveBayesClassifier(vocab_size=vocab_size)
      nb_classifier.fit(X_train, y_train)
      nb_accuracy = evaluate_classifier(nb_classifier, X_validate, y_validate)
      print(f'Naive Bayes Classifier Accuracy: {nb_accuracy:.4f}')
      
      lr_classifier = LogisticRegressionClassifier(
          l2=1e-4,
          random_state=42,
          draw_pictures=True,
          plot_dir=os.path.join('output', 'figures'),
          show_plots=False,
      )
      lr_classifier.fit(
          X_train,
          y_train,
          learning_rate=0.05,
          epochs=200,
          batch_size=64,
          lr_decay=0.01,
          class_weight='balanced',
          X_val=X_validate,
          y_val=y_validate,
          early_stopping_rounds=12,
      )
      lr_accuracy = evaluate_classifier(lr_classifier, X_validate, y_validate)
      print(f'Logistic Regression Classifier Accuracy: {lr_accuracy:.4f}')
      
      
if __name__ == "__main__":
    main()
      


