# feature_utils.py

import os
import re
import numpy as np
import pandas as pd
import emoji
# from textblob import TextBlob
# from langdetect import detect
# import language_tool_python
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split

# # === Basic text feature functions ===

# def count_language_errors(text):
#     try:
#         lang = detect(text)
#         lang_map = {'en': 'en-US', 'fr': 'fr', 'de': 'de', 'it': 'it', 'es': 'es', 'nl': 'nl'}
#         lang_code = lang_map.get(lang, 'en-US')
#         tool = language_tool_python.LanguageTool(lang_code)
#         matches = tool.check(text)
#         return len(matches)
#     except Exception:
#         return 0

# def get_toxicity_score(text):

#     from transformers import pipeline
#     # Load toxicity model once
#     toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", tokenizer="unitary/toxic-bert", truncation=True)

#     result = toxicity_model(text)
#     return result[0]['score']

# def get_sentiment(text):
#     return TextBlob(text).sentiment.polarity

# def count_words(text): return len(text.split())
# def count_links(text): return len(re.findall(r'https?://\S+', text))
# def count_mentions(text): return len(re.findall(r'@\w+', text))
# def extract_emojis(text): return [c for c in text if c in emoji.EMOJI_DATA]
# def has_question_mark(text): return int('?' in text)
# def has_exclamation_mark(text): return int('!' in text)
# def count_hashtags(text): return len(re.findall(r'#\w+', text))
# def count_punctuation(text): return len(re.findall(r'[.!?,;:]', text))
# def count_uppercase_letters(text): return sum(1 for c in text if c.isupper())
# def char_count(text): return len(text)
# def avg_word_length(text):
#     words = text.split()
#     return np.mean([len(w) for w in words]) if words else 0
# def has_quotes(text): return int(any(c in text for c in ['"', "'", '“', '”', '‘', '’']))

# # === Feature extraction ===

# def extract_features(df, cache_path=None):
#     expected_feature_funcs = {
#         'word_count': count_words,
#         'links_count': count_links,
#         'mentions_count': count_mentions,
#         'emojis_count': lambda s: len(extract_emojis(s)),
#         'avg_word_length': avg_word_length,
#         'punctuation_count': count_punctuation,
#         'uppercase_ratio': lambda s: count_uppercase_letters(s) / len(s) if len(s) > 0 else 0,
#         'hashtags_count': count_hashtags,
#         'has_question_mark': has_question_mark,
#         'has_exclamation_mark': has_exclamation_mark,
#         'has_link': None,  # derived
#         'has_mention': None,
#         'has_emoji': None,
#         'sentiment': get_sentiment,
#         'toxicity_score': get_toxicity_score,
#         'spelling_grammar_errors': count_language_errors,
#         'has_quotes': has_quotes,
#     }

#     if cache_path and os.path.exists(cache_path):
#         df_features = pd.read_csv(cache_path)
#         missing = [f for f in expected_feature_funcs if f not in df_features.columns]
#     else:
#         df_features = pd.DataFrame()
#         missing = list(expected_feature_funcs.keys())

#     for feature in missing:
#         if feature in ['has_link', 'has_mention', 'has_emoji']: continue
#         func = expected_feature_funcs[feature]
#         if func is not None:
#             df_features[feature] = df['text'].apply(func)

#     if 'has_link' not in df_features.columns and 'links_count' in df_features.columns:
#         df_features['has_link'] = (df_features['links_count'] > 0).astype(int)
#     if 'has_mention' not in df_features.columns and 'mentions_count' in df_features.columns:
#         df_features['has_mention'] = (df_features['mentions_count'] > 0).astype(int)
#     if 'has_emoji' not in df_features.columns and 'emojis_count' in df_features.columns:
#         df_features['has_emoji'] = (df_features['emojis_count'] > 0).astype(int)

        
#     if cache_path and missing:
#         df_features.to_csv(cache_path, index=False)

#     if cache_path:
#         merged_path = cache_path.replace(".csv", "_with_text_and_label.csv")
#         feature_df = df_features.copy()
#         feature_df['text'] = df['text'].values
#         feature_df['label'] = df['labels'].values
#         feature_df.to_csv(merged_path, index=False)

#     return df_features

# # === Model training and evaluation ===

# def evaluate_features_single_dataset(df, feature_cache_path=None, label_source="labels"):
#     assert label_source in df.columns, f"Label source '{label_source}' not found in DataFrame."
#     X = extract_features(df, cache_path=feature_cache_path)
#     y = df[label_source]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train)
#     y_pred_prob = clf.predict_proba(X_test)[:, 1]
#     auc = roc_auc_score(y_test, y_pred_prob)

#     # Feature importance
#     importances = pd.Series(clf.feature_importances_, index=X.columns)
    
#     # Initialize a list to store the correlation sign results
#     correlation_signs = []
    
#     # Analyze the sign of correlation for all features
#     for feature in X.columns:
#         # Calculate mean of the feature by label (0 vs. 1)
#         feature_by_label = pd.concat([X[feature], y], axis=1).groupby(label_source)[feature].mean()

#         # Determine the sign of the correlation based on means for each label
#         sign_of_correlation = "positive" if feature_by_label[1] > feature_by_label[0] else "negative"
        
#         # Store the feature's correlation sign along with its importance
#         correlation_signs.append({
#             "feature": feature,
#             "importance": importances[feature],
#             "correlation_sign": sign_of_correlation
#         })

#     # Convert the list of results into a DataFrame for easy viewing
#     correlation_df = pd.DataFrame(correlation_signs)

#     return auc, importances, correlation_df


# === Metadata parser ===

def parse_filename(filename):
    """
    Extracts model, finetuning, context, style, and OPPU info from a filename.
    Converts them into booleans or integers for clean tabular use.
    """
    base = os.path.basename(filename)
    parts = base.split("__")
    model = parts[0]
    ft = 1 if parts[1] == "ft" else 0
    context = 1 if parts[2] == "ctx1" else 0
    style = int(parts[3].replace("style", ""))
    oppu = 1 if parts[4].startswith("OPPU") else 0
    return model, ft, context, style, oppu

def make_label(model, ft, context, style, oppu):
    def bold(val, prefix):
        return f"{prefix}{val}" if val == 0 else rf"$\bf{{{prefix}{val}}}$"

    return "_".join([
        model,
        bold(style, "style"),
        bold(context, "ctx"),
        bold(ft, "ft"),
        bold(oppu, "oppu")
    ])

