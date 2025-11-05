# Provider Embedding Transformation Visualization

## Overview
This system allows you to visualize how a provider's raw procedure data transforms through each stage of the embedding pipeline, comparing two providers side-by-side.

## Files Created

### 1. `unified_embedding_pipeline_with_metadata.py`
**Updated training pipeline** with metadata saves for inference support.

**Changes from original:**
- Saves `specialty_code_indices` in Phase 1 metadata
- Saves `original_procedure_dim` in Phase 1 metadata  
- Saves same fields in final metadata for encoder initialization

**Run this first to train models.**

---

### 2. `procedure_encoder.py`
**Inference wrapper class** for encoding new providers.

**Key Features:**
- Loads all trained models (Phase 1, 3A, 3B)
- Preprocesses input (filters to specialty-relevant codes)
- Encodes through any stage of the pipeline
- Supports both sparse and dense input

**Usage:**
```python
from procedure_encoder import ProcedureEmbeddingEncoder

# Initialize
encoder = ProcedureEmbeddingEncoder(model_dir='./models')

# Encode new provider
new_provider_vector = load_procedure_vector()  # [num_codes]
embeddings = encoder.encode_full_pipeline(new_provider_vector)

# Access any stage
print(embeddings['agnostic'].shape)    # [1, latent_dim] - averaged
print(embeddings['multiview'].shape)   # [1, 1920] - 15 views
print(embeddings['reduced'].shape)     # [1, 256] - Phase 3A
print(embeddings['final'].shape)       # [1, 128] - Phase 3B

# Or encode to specific stage only
cancer_view = encoder.encode_phase1(new_provider_vector, specialty_id=0)
```

**Methods:**
- `encode_phase1(vector, specialty_id)` - Single specialty view
- `encode_specialty_agnostic(vector)` - Averaged across all specialties
- `encode_multiview(vector)` - All 15 specialty views concatenated
- `encode_reduced(multiview)` - Phase 3A reduction
- `encode_final(reduced)` - Phase 3B final compression
- `encode_full_pipeline(vector)` - Complete pipeline

---

### 3. `provider_transformation_visualization.py`
**Interactive Jupyter notebook** for comparing two providers.

**Required Data Files:**
```
procedure_vectors.npz                           # Raw procedure data
all_pins.npy                                    # Provider IDs
code_desc_df.parquet                            # Procedure code descriptions
procedure_df.parquet                            # Procedure claims per provider
phase2_multiview_embeddings.npy                 # Phase 2 outputs
phase3a_reduced_embeddings.npy                  # Phase 3A outputs
final_embeddings_128d.npy                       # Phase 3B outputs
final_pipeline_metadata.pkl                     # Metadata

Optional:
all_pin_names.parquet                           # Provider names
pin_to_label.pkl                                # Specialty labels
```

**Features:**
1. **Interactive Provider Selection**
   - Dropdown menus for Provider A and B
   - Searchable by name or PIN
   - Compare button

2. **Step 1: Raw Procedure Data**
   - Number of unique procedures
   - Total claims
   - Data statistics

3. **Step 2: Top 20 Procedures**
   - Side-by-side comparison
   - Procedure codes with descriptions
   - Claims and percentages
   - Common procedures highlighted

4. **Step 3: Specialty-Agnostic Embedding**
   - Averaged across all 15 specialties
   - Line plot comparison
   - Cosine similarity

5. **Step 4: Multi-View Embeddings** 
   - 15 specialty views concatenated (1920-dim)
   - Line plot with specialty labels on X-axis
   - Labels at midpoints of each view
   - Vertical lines separating specialties

6. **Step 5: Reduced Embeddings**
   - Phase 3A dimensionality reduction (256-dim)
   - Line plot comparison
   - Cosine similarity

7. **Step 6: Final Embeddings**
   - Phase 3B supervised compression (128-dim)
   - Line plot comparison
   - Cosine similarity + Euclidean distance

8. **Transformation Summary**
   - Similarity scores at each stage
   - Shows how similarity changes through pipeline

---

## Workflow

### Step 1: Train Pipeline
```bash
python unified_embedding_pipeline_with_metadata.py
```

**Outputs:**
- `phase1_specialty_conditioned_autoencoder.pth`
- `phase1_metadata.pkl`
- `phase2_multiview_embeddings.npy`
- `phase3a_reduction_autoencoder.pth`
- `phase3a_reduced_embeddings.npy`
- `phase3b_compression_network.pth`
- `final_embeddings_128d.npy`
- `final_pipeline_metadata.pkl`

### Step 2: Run Visualization
```bash
jupyter notebook provider_transformation_visualization.py
```

Or in Jupyter Lab:
1. Open `provider_transformation_visualization.py`
2. Run all cells
3. Use dropdown widgets to select providers
4. Click "Compare Providers"

---

## Architecture Summary

```
Raw Procedures [23,547 × num_codes]
    ↓ Filter to specialty-relevant codes
Filtered [23,547 × ~2,000]
    ↓ Phase 1: Specialty-conditioned autoencoder
Single Specialty View [23,547 × 128]
    ↓ Encode with all 15 specialties
Multi-View [23,547 × 1,920]  (15 × 128)
    ↓ Phase 3A: Reduction autoencoder
Reduced [23,547 × 256]
    ↓ Phase 3B: Supervised compression
Final [23,547 × 128]
```

---

## Key Concepts

### Specialty-Conditioned Encoding
- **One model** encodes from **any specialty perspective**
- Input: `[procedures + specialty_embedding]`
- Output: Specialty-specific latent representation
- Same provider → Different embeddings based on specialty view

### Multi-View Representation
- Provider encoded through **all 15 specialty lenses**
- Captures: "How does this provider look as Cancer? As Heart? As Rehab?"
- Concatenated into single 1,920-dim vector
- Preserves specialty-specific patterns

### Specialty-Agnostic View
- **Average** of all 15 specialty embeddings
- Gives "consensus" representation
- Useful for comparing providers without specialty bias

---

## Visualization Insights

**What to Look For:**

1. **Procedure Similarity**
   - Do providers share common high-volume procedures?
   - Different procedure mixes suggest different specialties

2. **Multi-View Patterns**
   - Which specialty views show highest values?
   - Spikes in specific specialty sections indicate focus areas

3. **Similarity Progression**
   - Does similarity increase or decrease through pipeline?
   - Final stage should cluster same-specialty providers

4. **Specialty Alignment**
   - Do labeled specialties match multi-view patterns?
   - Cancer hospital should spike in cancer view section

---

## Troubleshooting

**Error: "specialty_code_indices not found in metadata"**
- Run updated training pipeline first
- Old models don't have this field

**Error: "procedure_df.parquet not found"**
- Visualization needs raw procedure claims data
- Check data preparation steps

**Memory Issues:**
- Large procedure matrix may not fit in RAM
- Use sparse matrix operations
- Process in batches

**Slow Encoding:**
- GPU recommended for inference
- Batch multiple providers together
- Cache precomputed embeddings

---

## Extension Ideas

1. **Per-Specialty View Analysis**
   - Show all 15 specialty views separately
   - Heatmap across specialties

2. **Neighborhood Analysis**
   - Find K nearest neighbors at each stage
   - Show how neighborhoods change

3. **Trajectory Visualization**
   - 3D plot showing transformation path
   - UMAP/t-SNE at each stage

4. **Procedure Attribution**
   - Which procedures drive embedding values?
   - Attention weights visualization

5. **Batch Comparison**
   - Compare entire specialty groups
   - Distribution analysis

---

## Files Summary

| File | Purpose | Size | Required |
|------|---------|------|----------|
| unified_embedding_pipeline_with_metadata.py | Training pipeline | ~42KB | Training |
| procedure_encoder.py | Inference wrapper | ~15KB | Inference |
| provider_transformation_visualization.py | Interactive viz | ~15KB | Visualization |
| README_visualization.md | Documentation | ~8KB | Reference |

---

## Contact
For questions or issues, refer to the main project documentation.
