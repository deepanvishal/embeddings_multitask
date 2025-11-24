"""
GAT METRICS ANALYSIS - FINAL OPTIMIZED VERSION V2
==================================================
Computes pre-GAT similarity and attention metrics for provider pairs.
Includes new metric: alternative_claims_on_primary_top5_codes

Input:  all_providers_top10_alternatives_me2vec_county.csv
Output: all_providers_top10_alternatives_me2vec_county_WITH_GAT_METRICS.csv
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import load_npz
import gc
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GAT METRICS ANALYSIS - FINAL OPTIMIZED V2")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE_ATTENTION = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ATTENTION_CACHE_FILE = 'attention_cache.pkl'
print(f"\nDevice: {DEVICE}")

# ============================================================================
# STEP 1: LOAD ALL DATA FILES
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING DATA FILES")
print("="*80)

print("\nLoading pairs data from Code 2 output...")
pairs_df = pd.read_csv('all_providers_top10_alternatives_me2vec_county.csv')
print(f"  Loaded {len(pairs_df):,} pairs")

print("\nLoading embeddings...")
embeddings_df = pd.read_parquet('final_me2vec_all_towers_1046d.parquet')
emb_cols = [col for col in embeddings_df.columns if col != 'PIN']
final_embeddings_proc = embeddings_df[emb_cols].values[:, 0:512]
all_pins_list = embeddings_df['PIN'].values
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins_list)}
print(f"  Final embeddings (procedures): {final_embeddings_proc.shape}")

print("\nLoading initial embeddings...")
provider_init_embeddings = np.load('provider_init_embeddings.npy')
print(f"  Shape: {provider_init_embeddings.shape}")

print("\nLoading code embeddings...")
code_embeddings_dict = np.load('code_embeddings_dict.npy', allow_pickle=True).item()
code_ids_sorted = sorted(code_embeddings_dict.keys())
code_to_idx_map = {code: idx for idx, code in enumerate(code_ids_sorted)}
code_embeddings_array = np.array([code_embeddings_dict[code] for code in code_ids_sorted], dtype=np.float32)
print(f"  Code embeddings array: {code_embeddings_array.shape}")

print("\nLoading procedure matrix...")
proc_matrix = load_npz('procedure_vectors.npz')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

print("\nLoading mappings...")
with open('specialty_code_mappings.pkl', 'rb') as f:
    specialty_mappings = pickle.load(f)
with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

all_specialty_codes = sorted(list(set().union(*specialty_mappings['code_indices'].values())))
proc_matrix_filtered = proc_matrix[:, all_specialty_codes].tocsr()

print("\nLoading metadata...")
with open('me2vec_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

CODE_EMBEDDING_DIM = metadata['code_embedding_dim']
PROVIDER_EMBEDDING_DIM = metadata['provider_embedding_dim']
NUM_HEADS = metadata['num_heads']
num_specialties = metadata['num_specialties']

print(f"Model config: {CODE_EMBEDDING_DIM}D codes, {NUM_HEADS} heads, {num_specialties} specialties")

del code_embeddings_dict
gc.collect()

# ============================================================================
# STEP 2: DEFINE GAT MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 2: LOADING GAT MODEL")
print("="*80)

class ProviderGAT(nn.Module):
    def __init__(self, code_dim, hidden_dim, num_specialties, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        self.W_heads = nn.ModuleList([
            nn.Linear(code_dim, hidden_dim) for _ in range(num_heads)
        ])
        self.a_heads = nn.ParameterList([
            nn.Parameter(torch.randn(2 * hidden_dim, 1)) for _ in range(num_heads)
        ])
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * num_heads, num_specialties)
    
    def forward(self, provider_emb, code_embs):
        n_codes = code_embs.shape[0]
        attention_weights_all = []
        
        for head_idx in range(self.num_heads):
            W = self.W_heads[head_idx]
            a = self.a_heads[head_idx]
            
            provider_h = W(provider_emb)
            code_h = W(code_embs)
            
            provider_repeated = provider_h.repeat(n_codes, 1)
            concat = torch.cat([provider_repeated, code_h], dim=1)
            e = self.leaky_relu(concat @ a).squeeze()
            
            alpha = F.softmax(e, dim=0)
            attention_weights_all.append(alpha.detach().cpu().numpy())
        
        return np.array(attention_weights_all)

model = ProviderGAT(CODE_EMBEDDING_DIM, PROVIDER_EMBEDDING_DIM, 
                    num_specialties, NUM_HEADS, dropout=0.3).to(DEVICE)
model.load_state_dict(torch.load('gat_model.pth', map_location=DEVICE))
model.eval()
print("✓ Model loaded")

# ============================================================================
# STEP 3: COMPUTE PRE-GAT SIMILARITY (VECTORIZED)
# ============================================================================

print("\n" + "="*80)
print("STEP 3: COMPUTING PRE-GAT SIMILARITY")
print("="*80)

unique_primary = pairs_df['primary_pin'].unique()
unique_alternative = pairs_df['alternative_pin'].unique()
all_unique_pins = np.unique(np.concatenate([unique_primary, unique_alternative]))
print(f"Unique providers: {len(all_unique_pins):,}")

pairs_pin_to_idx = {pin: i for i, pin in enumerate(all_pins_list)}

print("\nMapping PINs to indices...")
primary_indices = np.zeros(len(pairs_df), dtype=np.int32)
alternative_indices = np.zeros(len(pairs_df), dtype=np.int32)

for idx in tqdm(range(len(pairs_df)), desc="Mapping"):
    p_pin = pairs_df.iloc[idx]['primary_pin']
    a_pin = pairs_df.iloc[idx]['alternative_pin']
    primary_indices[idx] = pairs_pin_to_idx.get(p_pin, -1)
    alternative_indices[idx] = pairs_pin_to_idx.get(a_pin, -1)

print("\nComputing pre-GAT similarities...")
valid_mask = (primary_indices >= 0) & (alternative_indices >= 0)
n_valid = valid_mask.sum()
pre_gat_similarities = np.zeros(len(pairs_df), dtype=np.float32)

if n_valid > 0:
    primary_embs = provider_init_embeddings[primary_indices[valid_mask]]
    alternative_embs = provider_init_embeddings[alternative_indices[valid_mask]]
    
    batch_size = 10000
    for i in tqdm(range(0, n_valid, batch_size), desc="Computing"):
        end_i = min(i + batch_size, n_valid)
        batch_primary = primary_embs[i:end_i]
        batch_alternative = alternative_embs[i:end_i]
        
        norms_p = np.linalg.norm(batch_primary, axis=1, keepdims=True)
        norms_a = np.linalg.norm(batch_alternative, axis=1, keepdims=True)
        norms_p = np.maximum(norms_p, 1e-8)
        norms_a = np.maximum(norms_a, 1e-8)
        
        batch_primary_norm = batch_primary / norms_p
        batch_alternative_norm = batch_alternative / norms_a
        batch_sims = (batch_primary_norm * batch_alternative_norm).sum(axis=1)
        
        valid_indices = np.where(valid_mask)[0][i:end_i]
        pre_gat_similarities[valid_indices] = batch_sims

pairs_df['pre_gat_similarity'] = pre_gat_similarities
pairs_df['similarity_delta'] = pairs_df['procedure_embedding_similarity'] - pairs_df['pre_gat_similarity']
pairs_df['similarity_change_pct'] = (pairs_df['similarity_delta'] / (pairs_df['pre_gat_similarity'] + 1e-8)) * 100

print(f"\nPre-GAT similarity: mean={pairs_df['pre_gat_similarity'].mean():.4f}")
print(f"Similarity delta: mean={pairs_df['similarity_delta'].mean():.4f}")

del primary_embs, alternative_embs, batch_primary, batch_alternative
gc.collect()

# ============================================================================
# STEP 4: EXTRACT ATTENTION (BATCHED + CACHED)
# ============================================================================

print("\n" + "="*80)
print("STEP 4: EXTRACTING ATTENTION WEIGHTS")
print("="*80)

if os.path.exists(ATTENTION_CACHE_FILE):
    print(f"✓ Loading cached attention: {ATTENTION_CACHE_FILE}")
    with open(ATTENTION_CACHE_FILE, 'rb') as f:
        attention_cache = pickle.load(f)
    print(f"  Loaded {len(attention_cache):,} providers")
else:
    print(f"Extracting attention for {len(all_unique_pins):,} providers...")
    attention_cache = {}
    pin_to_original_idx = {pin: idx for idx, pin in enumerate(all_pins)}
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(all_unique_pins), BATCH_SIZE_ATTENTION), desc="Extracting"):
            batch_end = min(batch_start + BATCH_SIZE_ATTENTION, len(all_unique_pins))
            batch_pins = all_unique_pins[batch_start:batch_end]
            
            for pin in batch_pins:
                if pin not in pin_to_original_idx:
                    continue
                
                provider_idx = pin_to_original_idx[pin]
                row = proc_matrix_filtered[provider_idx]
                
                if row.nnz == 0:
                    attention_cache[pin] = {'codes': [], 'attention': [], 'claims': []}
                    continue
                
                code_indices = row.indices
                code_counts = row.data
                
                code_list = []
                code_embs_list = []
                claim_counts = []
                
                for code_idx, count in zip(code_indices, code_counts):
                    code_id = all_specialty_codes[code_idx]
                    if code_id in code_to_idx_map:
                        code_list.append(code_id)
                        code_embs_list.append(code_embeddings_array[code_to_idx_map[code_id]])
                        claim_counts.append(count)
                
                if len(code_embs_list) == 0:
                    attention_cache[pin] = {'codes': [], 'attention': [], 'claims': []}
                    continue
                
                provider_emb = torch.FloatTensor(provider_init_embeddings[provider_idx]).unsqueeze(0).to(DEVICE)
                code_embs = torch.FloatTensor(np.array(code_embs_list, dtype=np.float32)).to(DEVICE)
                
                attention_weights = model(provider_emb, code_embs)
                avg_attention = attention_weights.mean(axis=0)
                
                attention_cache[pin] = {
                    'codes': code_list,
                    'attention': avg_attention.tolist(),
                    'claims': claim_counts
                }
    
    print(f"✓ Extracted attention for {len(attention_cache):,} providers")
    print(f"Saving cache...")
    with open(ATTENTION_CACHE_FILE, 'wb') as f:
        pickle.dump(attention_cache, f)
    print("✓ Cache saved")

# OPTIMIZATION: Create code-to-claims lookup for each provider
print("\nCreating code-to-claims lookup...")
provider_code_to_claims = {}
for pin, data in tqdm(attention_cache.items(), desc="Building lookups"):
    code_to_claims = {}
    for i, code in enumerate(data['codes']):
        if i < len(data['claims']):
            code_to_claims[code] = data['claims'][i]
    provider_code_to_claims[pin] = code_to_claims
print(f"✓ Created lookups for {len(provider_code_to_claims):,} providers")

# ============================================================================
# STEP 5: COMPUTE ATTENTION METRICS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: COMPUTING ATTENTION METRICS")
print("="*80)

print("\nComputing metrics...")

# Pre-allocate lists
attention_overlap_pct = []
num_shared_codes = []
A_top5_codes_list = []
B_top5_codes_list = []
A_top5_attention_list = []
B_top5_attention_list = []
A_top5_claims_list = []
B_top5_claims_list = []
alternative_claims_on_primary_top5_list = []  # NEW METRIC
shared_codes_count = []
A_only_codes_count = []
B_only_codes_count = []

for idx in tqdm(range(len(pairs_df)), desc="Computing"):
    p_pin = pairs_df.iloc[idx]['primary_pin']
    a_pin = pairs_df.iloc[idx]['alternative_pin']
    
    p_data = attention_cache.get(p_pin, {'codes': [], 'attention': [], 'claims': []})
    a_data = attention_cache.get(a_pin, {'codes': [], 'attention': [], 'claims': []})
    
    p_codes = p_data['codes'] if isinstance(p_data['codes'], list) else []
    a_codes = a_data['codes'] if isinstance(a_data['codes'], list) else []
    
    # Safe conversion to lists first, then numpy arrays
    p_attention_list = p_data['attention'] if isinstance(p_data['attention'], list) else []
    a_attention_list = a_data['attention'] if isinstance(a_data['attention'], list) else []
    p_attention = np.array(p_attention_list, dtype=np.float32) if len(p_attention_list) > 0 else np.array([], dtype=np.float32)
    a_attention = np.array(a_attention_list, dtype=np.float32) if len(a_attention_list) > 0 else np.array([], dtype=np.float32)
    
    p_claims = p_data['claims'] if isinstance(p_data['claims'], list) else []
    a_claims = a_data['claims'] if isinstance(a_data['claims'], list) else []
    
    # Primary's top 5 codes
    if len(p_codes) > 0 and len(p_attention_list) > 0 and p_attention.size > 0:
        if p_attention.size <= 5:
            p_top_indices = np.argsort(p_attention)[::-1]
        else:
            p_top_indices = np.argpartition(p_attention, -5)[-5:]
            p_top_indices = p_top_indices[np.argsort(p_attention[p_top_indices])[::-1]]
        
        p_top5_codes = [p_codes[i] for i in p_top_indices if i < len(p_codes)]
        p_top5_att = [float(p_attention[i]) for i in p_top_indices if i < p_attention.size]
        p_top5_clm = [p_claims[i] for i in p_top_indices if i < len(p_claims)]
    else:
        p_top5_codes = []
        p_top5_att = []
        p_top5_clm = []
    
    # Alternative's top 5 codes
    if len(a_codes) > 0 and len(a_attention_list) > 0 and a_attention.size > 0:
        if a_attention.size <= 5:
            a_top_indices = np.argsort(a_attention)[::-1]
        else:
            a_top_indices = np.argpartition(a_attention, -5)[-5:]
            a_top_indices = a_top_indices[np.argsort(a_attention[a_top_indices])[::-1]]
        
        a_top5_codes = [a_codes[i] for i in a_top_indices if i < len(a_codes)]
        a_top5_att = [float(a_attention[i]) for i in a_top_indices if i < a_attention.size]
        a_top5_clm = [a_claims[i] for i in a_top_indices if i < len(a_claims)]
    else:
        a_top5_codes = []
        a_top5_att = []
        a_top5_clm = []
    
    # NEW METRIC: Alternative's claims on primary's top 5 codes
    alternative_claims_on_primary_top5 = 0
    if len(p_top5_codes) > 0 and a_pin in provider_code_to_claims:
        a_code_to_claims = provider_code_to_claims[a_pin]
        for code in p_top5_codes:
            alternative_claims_on_primary_top5 += a_code_to_claims.get(code, 0)
    
    # Set operations
    p_codes_set = set(p_codes)
    a_codes_set = set(a_codes)
    shared = p_codes_set & a_codes_set
    p_only = p_codes_set - a_codes_set
    a_only = a_codes_set - p_codes_set
    
    # Attention overlap (top 10)
    if len(p_codes) > 0 and len(a_codes) > 0 and len(p_attention_list) > 0 and len(a_attention_list) > 0 and p_attention.size > 0 and a_attention.size > 0:
        if p_attention.size <= 10:
            p_top10_indices = np.argsort(p_attention)[::-1]
        else:
            p_top10_indices = np.argpartition(p_attention, -10)[-10:]
        
        if a_attention.size <= 10:
            a_top10_indices = np.argsort(a_attention)[::-1]
        else:
            a_top10_indices = np.argpartition(a_attention, -10)[-10:]
        
        p_top10_codes = set([p_codes[i] for i in p_top10_indices if i < len(p_codes)])
        a_top10_codes = set([a_codes[i] for i in a_top10_indices if i < len(a_codes)])
        
        overlap_codes = p_top10_codes & a_top10_codes
        union_codes = p_top10_codes | a_top10_codes
        overlap_pct = (len(overlap_codes) / len(union_codes) * 100) if len(union_codes) > 0 else 0.0
    else:
        overlap_pct = 0.0
    
    # Store results
    attention_overlap_pct.append(overlap_pct)
    num_shared_codes.append(len(shared))
    A_top5_codes_list.append(str(p_top5_codes))
    B_top5_codes_list.append(str(a_top5_codes))
    A_top5_attention_list.append(str([f"{x:.4f}" for x in p_top5_att]))
    B_top5_attention_list.append(str([f"{x:.4f}" for x in a_top5_att]))
    A_top5_claims_list.append(str(p_top5_clm))
    B_top5_claims_list.append(str(a_top5_clm))
    alternative_claims_on_primary_top5_list.append(alternative_claims_on_primary_top5)
    shared_codes_count.append(len(shared))
    A_only_codes_count.append(len(p_only))
    B_only_codes_count.append(len(a_only))

# Add to dataframe
pairs_df['attention_overlap_pct'] = attention_overlap_pct
pairs_df['num_shared_codes_total'] = num_shared_codes
pairs_df['primary_top5_codes'] = A_top5_codes_list
pairs_df['alternative_top5_codes'] = B_top5_codes_list
pairs_df['primary_top5_attention'] = A_top5_attention_list
pairs_df['alternative_top5_attention'] = B_top5_attention_list
pairs_df['primary_top5_claims'] = A_top5_claims_list
pairs_df['alternative_top5_claims'] = B_top5_claims_list
pairs_df['alternative_claims_on_primary_top5_codes'] = alternative_claims_on_primary_top5_list
pairs_df['shared_codes_count'] = shared_codes_count
pairs_df['primary_only_codes_count'] = A_only_codes_count
pairs_df['alternative_only_codes_count'] = B_only_codes_count

print(f"\n✓ Computed metrics for {len(pairs_df):,} pairs")
print(f"\nAttention overlap: mean={pairs_df['attention_overlap_pct'].mean():.2f}%")
print(f"Shared codes: mean={pairs_df['num_shared_codes_total'].mean():.1f}")
print(f"Alt claims on primary top5: mean={pairs_df['alternative_claims_on_primary_top5_codes'].mean():.1f}")

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: SAVING RESULTS")
print("="*80)

output_file = 'all_providers_top10_alternatives_me2vec_county_WITH_GAT_METRICS.csv'
pairs_df.to_csv(output_file, index=False)

print(f"\n✓ Saved: {output_file}")
print(f"  Rows: {len(pairs_df):,}")
print(f"  Columns: {len(pairs_df.columns)}")

print("\nNew columns added:")
new_cols = [
    'pre_gat_similarity',
    'similarity_delta',
    'similarity_change_pct',
    'attention_overlap_pct',
    'num_shared_codes_total',
    'primary_top5_codes',
    'alternative_top5_codes',
    'primary_top5_attention',
    'alternative_top5_attention',
    'primary_top5_claims',
    'alternative_top5_claims',
    'alternative_claims_on_primary_top5_codes',  # NEW
    'shared_codes_count',
    'primary_only_codes_count',
    'alternative_only_codes_count'
]
for col in new_cols:
    print(f"  - {col}")

# ============================================================================
# SUMMARY & VALIDATION
# ============================================================================

print("\n" + "="*80)
print("SUMMARY & VALIDATION")
print("="*80)

print("\nSimilarity Analysis:")
increased = (pairs_df['similarity_delta'] > 0).sum()
decreased = (pairs_df['similarity_delta'] < 0).sum()
print(f"  GAT increased similarity: {increased:,} ({increased/len(pairs_df):.1%})")
print(f"  GAT decreased similarity: {decreased:,} ({decreased/len(pairs_df):.1%})")

print("\nAttention Pattern Analysis:")
high_overlap = (pairs_df['attention_overlap_pct'] > 60).sum()
low_overlap = (pairs_df['attention_overlap_pct'] < 30).sum()
print(f"  High attention overlap (>60%): {high_overlap:,} pairs")
print(f"  Low attention overlap (<30%): {low_overlap:,} pairs")

print("\nNEW METRIC Validation:")
print(f"  Alternative claims on primary's top 5:")
print(f"    Mean: {pairs_df['alternative_claims_on_primary_top5_codes'].mean():.1f}")
print(f"    Median: {pairs_df['alternative_claims_on_primary_top5_codes'].median():.1f}")
print(f"    Min: {pairs_df['alternative_claims_on_primary_top5_codes'].min()}")
print(f"    Max: {pairs_df['alternative_claims_on_primary_top5_codes'].max()}")

print("\nCorrelation Analysis:")
corr_new_metric_sim = pairs_df[['alternative_claims_on_primary_top5_codes', 'procedure_embedding_similarity']].corr().iloc[0, 1]
corr_overlap_sim = pairs_df[['attention_overlap_pct', 'procedure_embedding_similarity']].corr().iloc[0, 1]
print(f"  Alt claims on primary top5 vs Post-GAT similarity: {corr_new_metric_sim:.3f}")
print(f"  Attention overlap vs Post-GAT similarity: {corr_overlap_sim:.3f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nKey metric for validation:")
print("  'alternative_claims_on_primary_top5_codes'")
print("    = Sum of alternative's claims on primary's top 5 attention codes")
print("\nTo validate: For each primary provider, rank alternatives by post-GAT")
print("similarity. The ranking should correlate with this new metric.")
