import json
import os
import pandas as pd
from pathlib import Path
import re

def sanitize_filename(filename):
    """
    Sanitize filename to be safe for filesystem
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    filename = filename[:50]
    return filename

def process_reddit_data(comments_file, submissions_file):
    """
    Process Reddit comments and submissions data
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
    # Main execution
    comments_file = '/scratch/nicpag/filtered_comments.jsonl'
    submissions_file = '/scratch/nicpag/rpolitics_submissions.txt'
    
    # Check if files exist
    if not os.path.exists(comments_file):
        print(f"Error: {comments_file} not found")
        exit(1)
    
    if not os.path.exists(submissions_file):
        print(f"Error: {submissions_file} not found")
        exit(1)
    
    try:
        process_reddit_data(comments_file, submissions_file)
    except Exception as e:
        print(f"An error occurred: {e}")
