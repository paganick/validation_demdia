"""
Parse raw Reddit data into structured CSV files organized by author.

This script processes JSONL files containing Reddit comments and submissions,
extracting comment-reply pairs and organizing them by author. It creates a
directory structure suitable for training personalized language models.

Input format:
    - Comments: JSONL file with one comment per line (filtered_comments.jsonl)
    - Submissions: JSONL file with one submission per line (rpolitics_submissions.txt)

Each comment contains:
    - author: Reddit username
    - body: Comment text
    - parent_id: ID of parent submission or comment (format: t3_xxxxx for submissions)

Output structure:
    reddit_data/
        {author1}/
            {author1}.csv (columns: text, reply_to)
        {author2}/
            {author2}.csv
        ...

The CSV files contain:
    - text: The comment body (AI will learn to generate this)
    - reply_to: The parent submission title or comment (context for generation)

This structure enables:
    1. Per-user datasets for personalized model training (OPPU)
    2. Easy filtering by author activity level
    3. Preservation of conversational context

Usage:
    python preprocessing/parse_reddit_data.py \\
        --comments filtered_comments.jsonl \\
        --submissions rpolitics_submissions.txt
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
import re

def sanitize_filename(filename):
    """
    Sanitize author names for safe filesystem usage.

    Removes invalid filename characters and limits length to prevent issues
    with long usernames.

    Args:
        filename: Author name or other string to use as filename

    Returns:
        str: Sanitized filename (max 50 characters, invalid chars replaced with _)
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    filename = filename[:50]
    return filename

def process_reddit_data(comments_file, submissions_file):
    """
    Extract comment-reply pairs from Reddit data and organize by author.

    Process flow:
        1. Load all submissions into memory for fast lookup
        2. Stream through comments line-by-line (handles large files)
        3. For each comment, find parent submission title
        4. Group comments by author into author-specific CSV files
        5. Append to existing CSVs to handle incremental processing

    Args:
        comments_file: Path to JSONL file with Reddit comments
        submissions_file: Path to JSONL file with Reddit submissions

    Side effects:
        Creates reddit_data/ directory with subdirectories per author,
        each containing {author}.csv with text and reply_to columns.

    Notes:
        - Only processes comments that reply to submissions (not comment replies)
        - Skips comments with missing author, body, or parent_id
        - Prints progress every 1000 comments
        - Handles malformed JSON gracefully by skipping bad lines
    """
    print("Loading submissions data...")
    
    # First, load all submissions into memory for quick lookup
    submissions_dict = {}
    
    try:
        with open(submissions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        submission = json.loads(line)
                        # Reddit submission IDs are prefixed with 't3_'
                        submission_id = submission.get('name', '')
                        if submission_id:
                            submissions_dict[submission_id] = submission.get('title', '')
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading submissions file: {e}")
        return
    
    print(f"Loaded {len(submissions_dict)} submissions")
    
    # Create main directory
    output_dir = Path('reddit_data')
    output_dir.mkdir(exist_ok=True)
    
    print("Processing comments...")
    
    # Process comments
    processed_count = 0
    with open(comments_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                comment = json.loads(line.strip())
                
                # Extract required fields
                author = comment.get('author', 'unknown')
                body = comment.get('body', '')
                parent_id = comment.get('parent_id', '')
                
                # Skip if essential data is missing
                if not author or not body or not parent_id:
                    continue
                
                # Find parent content
                parent_content = ''
                if parent_id in submissions_dict:
                    parent_content = submissions_dict[parent_id]
                else:
                    continue
                            
                # Create author directory
                safe_author = sanitize_filename(author)
                author_dir = output_dir / safe_author
                author_dir.mkdir(exist_ok=True)
                
                # Prepare data for CSV
                data = {
                    'text': [body],
                    'reply_to': [parent_content]
                }
                
                # Create DataFrame and save to CSV
                df = pd.DataFrame(data)
                csv_file = author_dir / f'{safe_author}.csv'
                
                # If file exists, append; otherwise create new
                if csv_file.exists():
                    existing_df = pd.read_csv(csv_file)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_csv(csv_file, index=False)
                else:
                    df.to_csv(csv_file, index=False)
                
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} comments...")
                    
            except json.JSONDecodeError:
                print(f"Error parsing JSON on line {line_num}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Processing complete! Processed {processed_count} comments")
    print(f"Data saved in '{output_dir}' directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse raw Reddit JSONL data into per-user post datasets."
    )
    parser.add_argument("--comments", required=True,
                        help="Path to filtered comments JSONL file")
    parser.add_argument("--submissions", required=True,
                        help="Path to Reddit submissions text file")
    args = parser.parse_args()

    for path in (args.comments, args.submissions):
        if not os.path.exists(path):
            print(f"Error: {path} not found")
            exit(1)

    try:
        process_reddit_data(args.comments, args.submissions)
    except Exception as e:
        print(f"An error occurred: {e}")
