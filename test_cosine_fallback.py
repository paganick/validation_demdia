#!/usr/bin/env python3
"""Quick test to verify consolidated data loader uses fallback for cosine similarities."""

import sys
sys.path.insert(0, 'plotting')

from consolidated_data_loader import load_cosine_similarities_by_response_type_consolidated

DATASETS = [
    'results/results_bluesky',
    'results/results_twitter',
    'results/results_reddit'
]

print("Testing consolidated cosine similarity loader...")
print("=" * 60)

similarity_data = load_cosine_similarities_by_response_type_consolidated(DATASETS)

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"Random records: {len(similarity_data['random'])}")
print(f"ML records: {len(similarity_data['ml'])}")
print(f"Cosine records: {len(similarity_data['cosine'])}")

if len(similarity_data['ml']) > 0 and len(similarity_data['cosine']) > 0:
    print("\n✅ SUCCESS: Cosine similarity data loaded (via fallback)")
else:
    print("\n❌ FAILED: No ML/cosine data loaded")
