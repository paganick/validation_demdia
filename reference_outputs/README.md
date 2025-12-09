# Reference Outputs for Reproducibility Testing

This directory contains reference outputs generated with fixed random seeds for reproducibility testing and verification.

## Purpose

These outputs serve as "golden" reference results to:
1. Verify that the simulation pipeline is working correctly after setup
2. Test reproducibility - running with the same seed should produce identical results
3. Provide a quick sanity check before running full-scale experiments

## Generation Details

**Model**: Llama-3.1-8B (base, not fine-tuned)
**Configuration**:
- No fine-tuning (`finetuned: false`)
- No context retrieval (`context_retrieval: false`)
- No style examples (`n_style_examples: 0`)
- With personas (`with_persona: true`)
- No OPPU (`OPPU: false`)
- Random seed: 42

**Dataset Sampling**:
- 1 user per dataset (randomly sampled with seed 42)
- 20 test messages per user
- 20 response candidates per message

## Files

- `meta-llama__Llama-3.1-8B__noft__ctx0__style0__no_OPPU__bluesky__random_response.json` - Bluesky dataset results
- `meta-llama__Llama-3.1-8B__noft__ctx0__style0__no_OPPU__twitter__random_response.json` - Twitter dataset results
- `meta-llama__Llama-3.1-8B__noft__ctx0__style0__no_OPPU__reddit__random_response.json` - Reddit dataset results

## Testing Reproducibility

To verify your setup produces identical results:

```bash
# Run simulations with the same parameters
python run_simulation.py \
    --config configs/llama3.1_base.yaml \
    --data_file data/bluesky/personas.pkl \
    --n_users 1 \
    --n_responses_per_user 20 \
    --seed 42 \
    --output_dir test_outputs

# Compare with reference outputs (should be identical)
diff reference_outputs/meta-llama/*.json test_outputs/meta-llama/*.json
```

If the outputs match, your setup is correctly configured and producing reproducible results.

## Notes

- These files are tracked in git despite being outputs (unlike `results/` directory)
- Total size: ~400KB (compressed outputs)
- Generation time: ~10-15 minutes total on NVIDIA A100
- Non-deterministic variations may occur across different hardware/CUDA versions, but results should be very similar

## Last Updated

Generated on: 2025-12-09
