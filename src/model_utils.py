from rank_bm25 import BM25Okapi

def build_bm25(history):
        tokenized = [doc.split() for doc in history]
        return BM25Okapi(tokenized)

def retrieve_context(prompt_text, bm25_client, history, k=3):
    query = prompt_text.split()
    top_docs = bm25_client.get_top_n(query, history, n=k)
    return "\n".join(top_docs)

## Help function for training personalized model

def format_conversation(row):
    # Plain prompt without any special tokens.
    return f"You are @{row['username']}. Respond to: '{row['reply_to']}'\n\n{row['message']}"

def preprocess_function(tokenizer, examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized