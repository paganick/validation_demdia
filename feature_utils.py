# feature_utils.py

import os
import re
import numpy as np
import pandas as pd
import emoji
import traceback
from collections import Counter
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from plotting_utils import parse_filename

# Load toxicity model once
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", tokenizer="unitary/toxic-bert", truncation=True, device=0)
sentiment_analyzer = SentimentIntensityAnalyzer()

# === Basic text feature functions ===

def count_language_errors(text):
    try:
        lang = detect(text)
        lang_map = {'en': 'en-US', 'fr': 'fr', 'de': 'de', 'it': 'it', 'es': 'es', 'nl': 'nl'}
        lang_code = lang_map.get(lang, 'en-US')
        tool = language_tool_python.LanguageTool(lang_code)
        matches = tool.check(text)
        return len(matches)
    except Exception:
        return 0

def get_toxicity_score_batch(texts, batch_size=20):
    results = toxicity_model(texts, batch_size=batch_size)
    return [r['score'] for r in results]

def get_toxicity_score(text):
    
    result = toxicity_model(text)
    return result[0]['score']

def get_sentiment_batch(texts):
    return [sentiment_analyzer.polarity_scores(t)['compound'] for t in texts]

def get_sentiment(text):
    return sentiment_analyzer.polarity_scores(text)['compound']

def count_words(text): return len(text.split())
def count_links(text): return len(re.findall(r'https?://\S+', text))
def count_mentions(text): return len(re.findall(r'@\w+', text))
def extract_emojis(text): return [c for c in text if c in emoji.EMOJI_DATA]
def has_question_mark(text): return int('?' in text)
def has_exclamation_mark(text): return int('!' in text)
def count_hashtags(text): return len(re.findall(r'#\w+', text))
def count_punctuation(text): return len(re.findall(r'[.!?,;:]', text))
def count_uppercase_letters(text): return sum(1 for c in text if c.isupper())
def char_count(text): return len(text)
def avg_word_length(text):
    words = text.split()
    return np.mean([len(w) for w in words]) if words else 0
def has_quotes(text): return int(any(c in text for c in ['"', "'", '“', '”', '‘', '’']))

# Download required NLTK data with fallbacks
def setup_nltk():
    """Setup NLTK with proper error handling and fallbacks"""
    required_downloads = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
    ]
    
    for resource_path, download_name in required_downloads:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                print(f"Downloading {download_name}...")
                nltk.download(download_name, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {download_name}: {e}")

setup_nltk()

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

# # Simple fallback tokenizers in case NLTK fails
# def simple_word_tokenize(text):
#     """Simple fallback word tokenizer"""
#     return re.findall(r'\b\w+\b', text.lower())

# def simple_sent_tokenize(text):
#     """Simple fallback sentence tokenizer"""
#     sentences = re.split(r'[.!?]+', text)
#     return [s.strip() for s in sentences if s.strip()]

# def safe_word_tokenize(text):
#     """Safe word tokenization with fallback"""
#     try:
#         return word_tokenize(str(text).lower())
#     except Exception:
#         return simple_word_tokenize(str(text))

# def safe_sent_tokenize(text):
#     """Safe sentence tokenization with fallback"""
#     try:
#         return sent_tokenize(str(text))
#     except Exception:
#         return simple_sent_tokenize(str(text))


# Load stopwords with fallback
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}


def count_hedge_words(text):
    """Count hedge words that indicate uncertainty"""
    hedge_words = {
        'perhaps', 'maybe', 'might', 'could', 'possibly', 'probably', 
        'likely', 'seems', 'appears', 'suggests', 'indicates', 'tend',
        'generally', 'usually', 'often', 'sometimes', 'fairly', 'rather',
        'somewhat', 'quite', 'relatively', 'approximately', 'roughly'
    }
    words = set(word_tokenize(text.lower()))
    return len(words.intersection(hedge_words))

def count_transition_words(text):
    """Count transitional phrases common in AI text"""
    transition_words = {
        'furthermore', 'moreover', 'however', 'nevertheless', 'nonetheless',
        'therefore', 'consequently', 'thus', 'hence', 'accordingly',
        'in conclusion', 'to summarize', 'in summary', 'overall',
        'additionally', 'furthermore', 'what is more', 'besides'
    }
    text_lower = text.lower()
    count = 0
    for phrase in transition_words:
        count += text_lower.count(phrase)
    return count

def count_superlatives(text):
    """Count superlative adjectives and adverbs"""
    superlative_words = {
        'best', 'worst', 'greatest', 'least', 'highest', 'lowest',
        'finest', 'largest', 'smallest', 'fastest', 'slowest', 'strongest',
        'weakest', 'brightest', 'darkest', 'richest', 'poorest'
    }
    
    words = word_tokenize(text.lower())
    counted_positions = set()  # Track which word positions we've already counted
    count = 0
    
    for i, word in enumerate(words):
        if i in counted_positions:
            continue
            
        # Check explicit superlative words (excluding 'most' for now)
        if word in superlative_words:
            count += 1
            counted_positions.add(i)
        # Check words ending in -est (but not already counted)
        elif word.endswith('est') and len(word) > 3:
            count += 1
            counted_positions.add(i)
        # Check "most + adjective" patterns
        elif word == 'most' and i + 1 < len(words):
            pos_tags = pos_tag([words[i + 1]])
            if pos_tags[0][1].startswith('JJ'):  # Adjective
                count += 1
                counted_positions.add(i)  # Only count the "most", not the adjective
    
    return count


def perplexity_proxy(text):
    """Simple perplexity proxy using word frequency"""
    words = word_tokenize(text.lower())
    if len(words) == 0:
        return 0
    
    # Count word frequencies
    word_counts = Counter(words)
    total_words = len(words)
    
    # Calculate entropy as perplexity proxy
    entropy = 0
    for count in word_counts.values():
        prob = count / total_words
        entropy -= prob * np.log2(prob) if prob > 0 else 0
    
    return entropy

def type_token_ratio(text):
    """Calculate type-token ratio (vocabulary diversity)"""
    words = word_tokenize(text.lower())
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)

def sentence_length_variance(text):
    """Calculate variance in sentence lengths"""
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return 0
    
    lengths = [len(word_tokenize(sent)) for sent in sentences]
    return np.var(lengths) if lengths else 0

def count_repetitive_patterns(text):
    """Count repetitive n-gram patterns"""
    words = word_tokenize(text.lower())
    if len(words) < 3:
        return 0
    
    # Count 3-gram repetitions
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    trigram_counts = Counter(trigrams)
    
    # Return number of trigrams that appear more than once
    return sum(1 for count in trigram_counts.values() if count > 1)

def abstract_concrete_ratio(text):
    """Calculate ratio of abstract to concrete words (simplified)"""
    abstract_indicators = {
        'concept', 'idea', 'theory', 'principle', 'notion', 'philosophy',
        'methodology', 'approach', 'framework', 'perspective', 'aspect',
        'factor', 'element', 'component', 'characteristic', 'feature',
        'quality', 'property', 'attribute', 'dimension', 'level'
    }
    
    concrete_indicators = {
        'person', 'people', 'man', 'woman', 'child', 'house', 'car',
        'tree', 'book', 'table', 'chair', 'door', 'window', 'phone',
        'computer', 'money', 'food', 'water', 'hand', 'face', 'eye'
    }
    
    words = set(word_tokenize(text.lower()))
    abstract_count = len(words.intersection(abstract_indicators))
    concrete_count = len(words.intersection(concrete_indicators))
    
    if concrete_count == 0:
        return abstract_count  # Avoid division by zero
    return abstract_count / concrete_count

def syntactic_complexity(text):
    """Simple syntactic complexity measure"""
    sentences = sent_tokenize(text)
    if not sentences:
        return 0
    
    # Count average clauses per sentence (approximated by comma count + 1)
    clause_counts = []
    for sent in sentences:
        # Simple approximation: count commas, semicolons, and conjunctions
        clause_markers = sent.count(',') + sent.count(';') + sent.count(' and ') + sent.count(' or ') + 1
        clause_counts.append(clause_markers)
    
    return np.mean(clause_counts) if clause_counts else 0


# Optimized versions that work with pre-tokenized data
def calculate_perplexity_proxy_from_tokens(tokens):
    """Calculate perplexity proxy from pre-tokenized text"""
    if len(tokens) == 0:
        return 0
    
    word_counts = Counter(tokens)
    total_words = len(tokens)
    
    entropy = 0
    for count in word_counts.values():
        prob = count / total_words
        entropy -= prob * np.log2(prob) if prob > 0 else 0
    
    return entropy

def calculate_ttr_from_tokens(tokens):
    """Calculate type-token ratio from pre-tokenized text"""
    if len(tokens) == 0:
        return 0
    return len(set(tokens)) / len(tokens)

def calculate_sentence_variance_from_splits(sentences):
    """Calculate sentence variance from pre-split sentences"""
    if len(sentences) <= 1:
        return 0
    
    try:
        lengths = [len(word_tokenize(str(sent))) for sent in sentences]
        return np.var(lengths) if lengths else 0
    except Exception:
        return 0

def count_repetitive_patterns_from_tokens(tokens):
    """Count repetitive patterns from pre-tokenized text"""
    if len(tokens) < 3:
        return 0
    
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    trigram_counts = Counter(trigrams)
    
    return sum(1 for count in trigram_counts.values() if count > 1)

def abstract_concrete_ratio_from_tokens(tokens):
    """Calculate abstract/concrete ratio from pre-tokenized text"""
    abstract_indicators = {
        'concept', 'idea', 'theory', 'principle', 'notion', 'philosophy',
        'methodology', 'approach', 'framework', 'perspective', 'aspect',
        'factor', 'element', 'component', 'characteristic', 'feature',
        'quality', 'property', 'attribute', 'dimension', 'level'
    }
    
    concrete_indicators = {
        'person', 'people', 'man', 'woman', 'child', 'house', 'car',
        'tree', 'book', 'table', 'chair', 'door', 'window', 'phone',
        'computer', 'money', 'food', 'water', 'hand', 'face', 'eye'
    }
    
    token_set = set(tokens)
    abstract_count = len(token_set.intersection(abstract_indicators))
    concrete_count = len(token_set.intersection(concrete_indicators))
    
    if concrete_count == 0:
        return abstract_count
    return abstract_count / concrete_count

def syntactic_complexity_from_sentences(sentences):
    """Calculate syntactic complexity from pre-split sentences"""
    if not sentences:
        return 0
    
    clause_counts = []
    for sent in sentences:
        clause_markers = sent.count(',') + sent.count(';') + sent.count(' and ') + sent.count(' or ') + 1
        clause_counts.append(clause_markers)
    
    return np.mean(clause_counts) if clause_counts else 0


# === Feature extraction ===

def extract_features(df, cache_path=None):
    expected_feature_funcs = {
    # === Basic features ===
    'word_count': count_words,
    'links_count': count_links,
    'mentions_count': count_mentions,
    'emojis_count': lambda s: len(extract_emojis(s)),
    'avg_word_length': avg_word_length,
    'punctuation_count': count_punctuation,
    'uppercase_ratio': lambda s: count_uppercase_letters(s) / len(s) if len(s) > 0 else 0,
    'hashtags_count': count_hashtags,
    'has_question_mark': has_question_mark,
    'has_exclamation_mark': has_exclamation_mark,
    'has_link': None,      # derived
    'has_mention': None,   # derived
    'has_emoji': None,     # derived
    'sentiment': 'batch_sentiment',
    'toxicity_score': 'batch_toxicity',
    'has_quotes': has_quotes,

    # === Advanced features ===
    'perplexity_proxy': perplexity_proxy,
    'type_token_ratio': type_token_ratio,
    'sentence_length_variance': sentence_length_variance,
    'hedge_words': count_hedge_words,
    'transition_words': count_transition_words,
    'superlatives': count_superlatives,
    'repetitive_patterns': count_repetitive_patterns,
    'abstract_concrete_ratio': abstract_concrete_ratio,
    'syntactic_complexity': syntactic_complexity,
    }


    texts = df['text'].fillna('').astype(str).tolist()

    if cache_path and os.path.exists(cache_path):
        df_features = pd.read_csv(cache_path)
        missing = [f for f in expected_feature_funcs if f not in df_features.columns]
    else:
        df_features = pd.DataFrame()
        missing = list(expected_feature_funcs.keys())

    for feature in missing:
        if feature in ['has_link', 'has_mention', 'has_emoji']:
            continue

        func = expected_feature_funcs[feature]
        
        # Batch sentiment
        if func == 'batch_sentiment':
            df_features['sentiment'] = get_sentiment_batch(texts)
        
        # Batch toxicity
        elif func == 'batch_toxicity':
            df_features['toxicity_score'] = get_toxicity_score_batch(texts)
        
        # Regular apply
        elif func is not None:
            df_features[feature] = df['text'].apply(func)

    # Derived boolean features
    if 'has_link' not in df_features.columns and 'links_count' in df_features.columns:
        df_features['has_link'] = (df_features['links_count'] > 0).astype(int)
    if 'has_mention' not in df_features.columns and 'mentions_count' in df_features.columns:
        df_features['has_mention'] = (df_features['mentions_count'] > 0).astype(int)
    if 'has_emoji' not in df_features.columns and 'emojis_count' in df_features.columns:
        df_features['has_emoji'] = (df_features['emojis_count'] > 0).astype(int)

    if cache_path and missing:
        df_features.to_csv(cache_path, index=False)

    if cache_path:
        merged_path = cache_path.replace(".csv", "_with_text_and_label.csv")
        feature_df = df_features.copy()
        feature_df['text'] = df['text'].values
        feature_df['label'] = df['labels'].values
        feature_df.to_csv(merged_path, index=False)

    return df_features

# === Model training and evaluation ===

def evaluate_features_single_dataset(
    df, 
    feature_cache_path=None, 
    label_source="labels", 
    exclude_features=['spelling_grammar_errors', 'has_emoji', 'has_mention', 'has_link']
):
    assert label_source in df.columns, f"Label source '{label_source}' not found in DataFrame."
    exclude_features = exclude_features or []
    

    print("Class distribution:", df[label_source].value_counts())

    X = extract_features(df, cache_path=feature_cache_path)
    y = df[label_source]

    # Drop excluded features if they exist in the DataFrame
    X = X.drop(columns=[f for f in exclude_features if f in X.columns], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=X.columns)

    # Analyze the sign of correlation for all remaining features
    correlation_signs = []
    for feature in X.columns:
        feature_by_label = pd.concat([X[feature], y], axis=1).groupby(label_source)[feature].mean()
        sign_of_correlation = "positive" if feature_by_label[1] > feature_by_label[0] else "negative"
        correlation_signs.append({
            "feature": feature,
            "importance": importances[feature],
            "correlation_sign": sign_of_correlation
        })

    correlation_df = pd.DataFrame(correlation_signs)

    return auc, importances, correlation_df

def safe_read_csv(file_path, max_retries=3):
    """
    Safely read CSV with multiple fallback strategies for malformed files.
    """
    print(f"DEBUG: Attempting to read CSV: {file_path}")
    
    # Check if file exists and get basic info
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist: {file_path}")
        return None
        
    file_size = os.path.getsize(file_path)
    print(f"DEBUG: File size: {file_size:,} bytes")
    
    # Strategy 1: Try default pandas read_csv
    try:
        print("DEBUG: Trying default pandas read_csv...")
        df = pd.read_csv(file_path)
        print(f"DEBUG: Successfully loaded with default method. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"DEBUG: Default method failed: {str(e)[:200]}")
    
    # Strategy 2: Try with error handling and different engine
    try:
        print("DEBUG: Trying with python engine and error handling...")
        df = pd.read_csv(file_path, engine='python', on_bad_lines='warn')
        print(f"DEBUG: Successfully loaded with python engine. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"DEBUG: Python engine method failed: {str(e)[:200]}")
    
    print("ERROR: All CSV reading strategies failed. File may be severely corrupted.")
    return None

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

def find_median_accuracy_run(trainer_results_path):
    """
    Find the training run with median accuracy from BERT trainer results.
    
    Args:
        trainer_results_path: Path to the JSON file containing training results
        
    Returns:
        dict: The training run data with median accuracy
    """
    print(f"DEBUG: Loading trainer results from: {trainer_results_path}")
    
    with open(trainer_results_path, 'r') as f:
        results = json.load(f)
    
    # Extract accuracies and find median
    accuracies = [run['accuracy'] for run in results]
    print(f"DEBUG: Found accuracies: {accuracies}")
    
    median_accuracy = np.median(accuracies)
    print(f"DEBUG: Median accuracy: {median_accuracy}")
    
    # Find the run closest to median accuracy
    median_run = min(results, key=lambda x: abs(x['accuracy'] - median_accuracy))
    print(f"DEBUG: Selected run {median_run['run']} with accuracy {median_run['accuracy']}")
    
    return median_run

def create_validation_dataframe(median_run):
    """
    Create a DataFrame from the median run's validation data.
    
    Args:
        median_run: Dictionary containing the median accuracy run data
        
    Returns:
        pd.DataFrame: DataFrame with text, labels, and bert_prediction columns
    """
    val_texts = median_run['val_texts']
    true_labels = median_run['true_labels']
    bert_predictions = median_run['predictions']
    
    print(f"DEBUG: Creating DataFrame with {len(val_texts)} validation samples")
    
    df = pd.DataFrame({
        'text': val_texts,
        'labels': true_labels,
        'bert_prediction': bert_predictions
    })
    
    print(f"DEBUG: Validation DataFrame shape: {df.shape}")
    print(f"DEBUG: Label distribution - True: {pd.Series(true_labels).value_counts().to_dict()}")
    print(f"DEBUG: Label distribution - BERT: {pd.Series(bert_predictions).value_counts().to_dict()}")
    
    return df

def evaluate_with_median_run_data(full_path, label_source):
    """
    Modified evaluation function that uses median accuracy run validation data.
    Trains Random Forest on all data EXCEPT the validation set.
    
    Args:
        full_path: Path to the original validation data file
        label_source: Either 'labels' or 'bert_prediction'
    """
    print(f"DEBUG: Starting evaluation with median run data for: {full_path}")
    
    # Find corresponding trainer results file
    trainer_results_path = full_path.replace('_validation_data_labelled.csv', '_validation_data_trainer_results.json')
    
    if not os.path.exists(trainer_results_path):
        print(f"ERROR: Trainer results file not found: {trainer_results_path}")
        return
    
    # Get median accuracy run data
    median_run = find_median_accuracy_run(trainer_results_path)
    
    # Create validation DataFrame
    val_df = create_validation_dataframe(median_run)
    
    # Load full dataset 
    print(f"DEBUG: Loading full dataset...")
    full_df = safe_read_csv(full_path)
    if full_df is None:
        print(f"ERROR: Could not read full dataset: {full_path}")
        return
    
    print(f"DEBUG: Full dataset shape: {full_df.shape}")
    
    # Clean the full dataset
    full_df['text'] = full_df['text'].fillna('').astype(str)
    full_df = full_df.dropna(subset=[label_source])
    
    print(f"DEBUG: Full dataset after cleaning: {full_df.shape}")
    
    # Create a set of validation texts to exclude from training
    val_texts_set = set(val_df['text'].tolist())
    print(f"DEBUG: Validation set contains {len(val_texts_set)} unique texts")
    
    # Split full dataset: training = all data EXCEPT validation texts
    train_mask = ~full_df['text'].isin(val_texts_set)
    train_df = full_df[train_mask].copy()
    
    print(f"DEBUG: Training set shape after excluding validation: {train_df.shape}")
    print(f"DEBUG: Excluded {len(full_df) - len(train_df)} validation samples from training")
    
    if len(train_df) == 0:
        print(f"ERROR: No training data remaining after excluding validation set!")
        return
    
    # Extract features for training dataset
    feature_cache_path = full_path.replace("_labelled.csv", "_features.csv")
    print(f"DEBUG: Using feature cache: {feature_cache_path}")
    
    if not os.path.exists(feature_cache_path):
        print(f"ERROR: Feature cache not found: {feature_cache_path}")
        print("Run 'compute_features' first!")
        return
    
    # Load full feature cache
    features_df = pd.read_csv(feature_cache_path)
    print(f"DEBUG: Full features shape: {features_df.shape}")
    
    if len(features_df) != len(full_df):
        print(f"WARNING: Feature cache size mismatch. Regenerating features...")
        features_df = extract_features(full_df, cache_path=feature_cache_path)
    
    # IMPORTANT: Reset indices to ensure alignment
    full_df = full_df.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    
    # Verify alignment after reset
    if len(features_df) != len(full_df):
        print(f"ERROR: After reset, features ({len(features_df)}) and data ({len(full_df)}) still don't match!")
        return
    
    print(f"DEBUG: After index reset - Full DF: {full_df.shape}, Features: {features_df.shape}")
    
    # Recreate train_mask after index reset
    train_mask = ~full_df['text'].isin(val_texts_set)
    train_df = full_df[train_mask].copy()
    
    print(f"DEBUG: Training set shape after excluding validation: {train_df.shape}")
    print(f"DEBUG: Excluded {len(full_df) - len(train_df)} validation samples from training")
    
    # Get features for training set only (exclude validation indices)
    train_features = features_df[train_mask].copy()
    print(f"DEBUG: Training features shape: {train_features.shape}")
    
    # Verify that train_features and train_df have matching lengths
    if len(train_features) != len(train_df):
        print(f"ERROR: Training features ({len(train_features)}) and training data ({len(train_df)}) lengths don't match!")
        return
    
    # Train Random Forest on training set only
    print(f"DEBUG: Training Random Forest on training set (excluding validation)...")
    exclude_features = ['spelling_grammar_errors', 'has_emoji', 'has_mention', 'has_link']
    X_train = train_features.drop(columns=[f for f in exclude_features if f in train_features.columns], errors="ignore")
    y_train = train_df[label_source]
    
    print(f"DEBUG: Training set label distribution: {y_train.value_counts().to_dict()}")
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    print(f"DEBUG: Random Forest trained successfully on {len(X_train)} samples")
    
    # Extract features for validation set
    print(f"DEBUG: Extracting features for validation set...")
    val_features = extract_features(val_df)
    X_val = val_features.drop(columns=[f for f in exclude_features if f in val_features.columns], errors="ignore")
    
    # Ensure same feature columns
    missing_features = set(X_train.columns) - set(X_val.columns)
    extra_features = set(X_val.columns) - set(X_train.columns)
    
    if missing_features:
        print(f"WARNING: Missing features in validation set: {missing_features}")
        for feature in missing_features:
            X_val[feature] = 0  # Add missing features with default value
    
    if extra_features:
        print(f"WARNING: Extra features in validation set: {extra_features}")
        X_val = X_val.drop(columns=list(extra_features))
    
    # Reorder columns to match training set
    X_val = X_val[X_train.columns]
    
    # Make predictions on validation set
    print(f"DEBUG: Making Random Forest predictions on validation set...")
    rf_predictions = clf.predict(X_val)
    rf_probabilities = clf.predict_proba(X_val)[:, 1]
    
    # Calculate AUC
    y_val = val_df[label_source]
    auc = roc_auc_score(y_val, rf_probabilities)
    print(f"DEBUG: Random Forest AUC on validation set: {auc:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'text': val_df['text'],
        'actual_label': val_df['labels'],
        'bert_prediction': val_df['bert_prediction'],
        'rf_prediction': rf_predictions,
        'rf_probability': rf_probabilities
    })
    
    # Save results
    suffix = "_from_bert" if label_source == "bert_prediction" else "_from_labels"
    output_path = full_path.replace("_labelled.csv", f"{suffix}_median_run_results.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"DEBUG: Results saved to: {output_path}")
    print(f"DEBUG: Results shape: {results_df.shape}")
    
    # Print summary statistics
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Validation set size: {len(results_df)}")
    print(f"Random Forest AUC: {auc:.4f}")
    print(f"Actual label distribution: {results_df['actual_label'].value_counts().to_dict()}")
    print(f"BERT prediction distribution: {results_df['bert_prediction'].value_counts().to_dict()}")
    print(f"RF prediction distribution: {results_df['rf_prediction'].value_counts().to_dict()}")
    
    # Calculate agreement between models
    bert_rf_agreement = (results_df['bert_prediction'] == results_df['rf_prediction']).mean()
    bert_actual_agreement = (results_df['bert_prediction'] == results_df['actual_label']).mean()
    rf_actual_agreement = (results_df['rf_prediction'] == results_df['actual_label']).mean()
    
    # Three-way agreement analysis
    all_three_agree = (
        (results_df['actual_label'] == results_df['bert_prediction']) & 
        (results_df['bert_prediction'] == results_df['rf_prediction'])
    ).mean()
    
    # Models agree with each other (but may be wrong)
    models_agree = (results_df['bert_prediction'] == results_df['rf_prediction']).mean()
    
    # Models agree AND are correct
    models_agree_correct = (
        (results_df['bert_prediction'] == results_df['rf_prediction']) & 
        (results_df['bert_prediction'] == results_df['actual_label'])
    ).mean()
    
    # Models agree BUT are wrong
    models_agree_wrong = (
        (results_df['bert_prediction'] == results_df['rf_prediction']) & 
        (results_df['bert_prediction'] != results_df['actual_label'])
    ).mean()
    
    # Majority vote analysis (2 out of 3 agree)
    # BERT and Actual agree (RF disagrees)
    bert_actual_vs_rf = (
        (results_df['bert_prediction'] == results_df['actual_label']) & 
        (results_df['rf_prediction'] != results_df['actual_label'])
    ).mean()
    
    # RF and Actual agree (BERT disagrees)  
    rf_actual_vs_bert = (
        (results_df['rf_prediction'] == results_df['actual_label']) & 
        (results_df['bert_prediction'] != results_df['actual_label'])
    ).mean()
    
    # Both models wrong, but in same way
    both_models_wrong_same = (
        (results_df['bert_prediction'] == results_df['rf_prediction']) & 
        (results_df['bert_prediction'] != results_df['actual_label'])
    ).mean()
    
    # Both models wrong, but in different ways
    both_models_wrong_different = (
        (results_df['bert_prediction'] != results_df['actual_label']) & 
        (results_df['rf_prediction'] != results_df['actual_label']) &
        (results_df['bert_prediction'] != results_df['rf_prediction'])
    ).mean()
    
    print(f"BERT-RF agreement: {bert_rf_agreement:.4f}")
    print(f"BERT-Actual agreement: {bert_actual_agreement:.4f}")
    print(f"RF-Actual agreement: {rf_actual_agreement:.4f}")
    print(f"All three agree: {all_three_agree:.4f}")
    print(f"Models agree (right or wrong): {models_agree:.4f}")
    print(f"Models agree AND correct: {models_agree_correct:.4f}")
    print(f"Models agree BUT wrong: {models_agree_wrong:.4f}")
    
    # Create comprehensive agreement statistics DataFrame
    agreement_stats = pd.DataFrame({
        'metric': [
            'bert_rf_agreement', 'bert_actual_agreement', 'rf_actual_agreement', 
            'all_three_agree', 'models_agree', 'models_agree_correct', 'models_agree_wrong',
            'bert_actual_vs_rf', 'rf_actual_vs_bert', 'both_models_wrong_same', 
            'both_models_wrong_different', 'rf_auc', 'validation_size'
        ],
        'value': [
            bert_rf_agreement, bert_actual_agreement, rf_actual_agreement,
            all_three_agree, models_agree, models_agree_correct, models_agree_wrong,
            bert_actual_vs_rf, rf_actual_vs_bert, both_models_wrong_same,
            both_models_wrong_different, auc, len(results_df)
        ]
    })
    
    # Save agreement statistics
    agreement_output_path = full_path.replace("_labelled.csv", f"{suffix}_median_run_agreement.csv")
    agreement_stats.to_csv(agreement_output_path, index=False)
    print(f"DEBUG: Agreement statistics saved to: {agreement_output_path}")
    
    return auc, results_df, agreement_stats

def evaluate_all_datasets_median_run(folder_path, label_source):
    """
    Modified version of evaluate_all_datasets that uses median run validation data.
    """
    print(f"DEBUG: Starting evaluate_all_datasets_median_run with folder_path: {folder_path}, label_source: {label_source}")
    
    assert label_source in ['labels', 'bert_prediction'], "label_source must be 'labels' or 'bert_prediction'"

    if not os.path.exists(folder_path):
        print(f"ERROR: Folder path does not exist: {folder_path}")
        return

    results_summary = []
    agreement_summary = []
    files_found = 0
    files_processed = 0

    for root, _, files in os.walk(folder_path):
        print(f"DEBUG: Scanning directory: {root}")
        
        for filename in files:
            if filename.endswith('validation_data_labelled.csv') and ('random' in filename or 'cosine' in filename or 'ml' in filename):
                files_found += 1
                full_path = os.path.join(root, filename)
                print(f'DEBUG: Found matching file #{files_found}: {full_path}')

                try:
                    auc, results_df, agreement_stats = evaluate_with_median_run_data(full_path, label_source)
                    
                    # Parse filename for metadata
                    try:
                        model, ft, context, style, oppu = parse_filename(filename)
                        print(f"DEBUG: Parsed filename - model: {model}, ft: {ft}, context: {context}, style: {style}, oppu: {oppu}")
                        if 'random' in filename:
                            source_type = 'random'
                        else:
                            source_type = "cosine" if "cosine" in filename else "ml"
                        # Collect summary results
                        results_summary.append({
                            'model': model,
                            'ft': ft,
                            'context': context,
                            'style': style,
                            'oppu': oppu,
                            'auc': auc,
                            'label_source': label_source,
                            'source_type': source_type,
                            'validation_size': len(results_df)
                        })
                        
                        # Collect agreement statistics with metadata
                        for _, row in agreement_stats.iterrows():
                            agreement_summary.append({
                                'model': model,
                                'ft': ft,
                                'context': context,
                                'style': style,
                                'oppu': oppu,
                                'label_source': label_source,
                                'source_type': source_type,
                                'metric': row['metric'],
                                'value': row['value']
                            })
                        
                    except Exception as parse_error:
                        print(f"ERROR: Could not parse filename {filename}: {parse_error}")
                        continue
                        
                    files_processed += 1
                    
                except Exception as e:
                    print(f"ERROR: Failed to evaluate {full_path}")
                    print(f"ERROR: {str(e)}")
                    print("Full traceback:")
                    traceback.print_exc()
                    continue

    print(f"DEBUG: Summary - Found {files_found} files, successfully processed {files_processed}")
    
    if results_summary:
        # Save combined summary
        suffix = "_from_bert" if label_source == "bert_prediction" else "_from_labels"
        
        # Save evaluation summary
        summary_path = os.path.join(folder_path, f'median_run_evaluation_summary{suffix}.csv')
        pd.DataFrame(results_summary).to_csv(summary_path, index=False)
        print(f"DEBUG: Evaluation summary saved to: {summary_path}")
        
        # Save agreement summary
        agreement_summary_path = os.path.join(folder_path, f'median_run_agreement_summary{suffix}.csv')
        pd.DataFrame(agreement_summary).to_csv(agreement_summary_path, index=False)
        print(f"DEBUG: Agreement summary saved to: {agreement_summary_path}")
        
        # Print overall statistics
        print(f"\n=== OVERALL SUMMARY ===")
        print(f"Processed {files_processed} datasets")
        
        if agreement_summary:
            agreement_df = pd.DataFrame(agreement_summary)
            
            # Calculate average agreements across all datasets
            bert_rf_avg = agreement_df[agreement_df['metric'] == 'bert_rf_agreement']['value'].mean()
            bert_actual_avg = agreement_df[agreement_df['metric'] == 'bert_actual_agreement']['value'].mean()
            rf_actual_avg = agreement_df[agreement_df['metric'] == 'rf_actual_agreement']['value'].mean()
            rf_auc_avg = agreement_df[agreement_df['metric'] == 'rf_auc']['value'].mean()
            
            print(f"Average BERT-RF agreement: {bert_rf_avg:.4f}")
            print(f"Average BERT-Actual agreement: {bert_actual_avg:.4f}")
            print(f"Average RF-Actual agreement: {rf_actual_avg:.4f}")
            print(f"Average RF AUC: {rf_auc_avg:.4f}")