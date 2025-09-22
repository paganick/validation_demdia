# feature_utils.py

import os
import re
import numpy as np
import pandas as pd
import emoji
from collections import Counter
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
