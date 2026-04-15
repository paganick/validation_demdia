from rank_bm25 import BM25Okapi

def build_bm25(history):
    """Build a BM25 index over a list of document strings."""
    tokenized = [doc.split() for doc in history]
    return BM25Okapi(tokenized)

def retrieve_context(prompt_text, bm25_client, history, k=3):
    """Return the top-k most relevant documents from history for the given prompt."""
    query = prompt_text.split()
    top_docs = bm25_client.get_top_n(query, history, n=k)
    return "\n".join(top_docs)

def format_conversation(row):
    """Format a training row as a plain instruction-response string for fine-tuning."""
    # Plain prompt without any special tokens.
    return f"You are @{row['username']}. Respond to: '{row['reply_to']}'\n\n{row['message']}"

def preprocess_function(tokenizer, examples):
    """Tokenize examples for causal-LM training (input_ids == labels)."""
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized