"""
TOP 10 ALTERNATIVES FOR ALL PROVIDERS - ME2VEC + CNP
====================================================

Uses CNP (Conditional Neighbor Probability) for directional similarity.
Includes claims-based overlap metrics for procedures and diagnoses.

CNP(A→B): "Can B substitute for A?" (used for ranking)
CNP(B→A): "Can A substitute for B?" (for validation)

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import gc

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# ============================================================================
# LOAD ALL DATA FILES
# ============================================================================

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)
print(f"Labels: {len(pin_to_label)}")

embeddings_df = pd.read_parquet('final_me2vec_all_towers_1046d.parquet')
print(f"Embeddings: {embeddings_df.shape}")

procedure_df = pd.read_parquet('procedure_df.parquet')
print(f"Procedure data: {procedure_df.shape}")

diagnosis_df = pd.read_parquet('diagnosis_df.parquet')
print(f"Diagnosis data: {diagnosis_df.shape}")

demo_df = pd.read_parquet('demo_df.parquet')
print(f"Demographics: {demo_df.shape}")

place_df = pd.read_parquet('place_df.parquet')
print(f"Place: {place_df.shape}")

cost_df = pd.read_parquet('cost_df.parquet')
print(f"Cost: {cost_df.shape}")

pin_df = pd.read_parquet('pin_df.parquet')
print(f"PIN summary: {pin_df.shape}")

pin_names_df = pd.read_parquet('all_pin_names.parquet')
all_pins_with_embeddings = embeddings_df['PIN'].values
pin_names_df = pin_names_df[pin_names_df['PIN'].isin(all_pins_with_embeddings)]
print(f"PIN names: {pin_names_df.shape}")

pin_to_name = dict(zip(pin_names_df['PIN'], pin_names_df['PIN_name']))

prov_spl_df = pd.read_parquet('prov_spl.parquet')
print(f"Provider specialty: {prov_spl_df.shape}")

pin_to_specialty = dict(zip(prov_spl_df['PIN'], prov_spl_df['srv_spclty_ctg_cd']))
print(f"PIN to specialty mapping: {len(pin_to_specialty)} providers")

print("\n" + "="*80)
print("LOADING CNP MATRIX")
print("="*80)

cnp_matrix = np.load('cnp_matrix.npy')
print(f"CNP matrix shape: {cnp_matrix.shape}")
print(f"CNP matrix stats: min={cnp_matrix.min():.6f}, max={cnp_matrix.max():.6f}, mean={cnp_matrix.mean():.6f}")

# ============================================================================
# PREPARE DATA STRUCTURES
# ============================================================================

print("\n" + "="*80)
print("PREPARING DATA STRUCTURES")
print("="*80)

all_pins_list = embeddings_df['PIN'].values.tolist()
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins_list)}

emb_cols = [col for col in embeddings_df.columns if col != 'PIN']
pin_to_emb = {}
for _, row in embeddings_df.iterrows():
    pin = row['PIN']
    emb = row[emb_cols].values
    pin_to_emb[pin] = emb

print("Processing procedure data with claims...")
procedure_summary = procedure_df.groupby('PIN').agg({
    'code': lambda x: set(x),
    'claims': lambda x: dict(zip(procedure_df.loc[x.index, 'code'], x))
}).reset_index()
procedure_summary.columns = ['PIN', 'codes', 'code_to_claims']

pin_to_procedure_codes = dict(zip(procedure_summary['PIN'], procedure_summary['codes']))
pin_to_procedure_claims = dict(zip(procedure_summary['PIN'], procedure_summary['code_to_claims']))

print("Processing diagnosis data with claims...")
diagnosis_summary = diagnosis_df.groupby('PIN').agg({
    'code': lambda x: set(x),
    'claims': lambda x: dict(zip(diagnosis_df.loc[x.index, 'code'], x))
}).reset_index()
diagnosis_summary.columns = ['PIN', 'codes', 'code_to_claims']

pin_to_diagnosis_codes = dict(zip(diagnosis_summary['PIN'], diagnosis_summary['codes']))
pin_to_diagnosis_claims = dict(zip(diagnosis_summary['PIN'], diagnosis_summary['code_to_claims']))

demo_df = demo_df.set_index('PIN')
place_df = place_df.set_index('PIN')
cost_df = cost_df.set_index('PIN')
pin_df = pin_df.set_index('PIN')

demo_cols = [col for col in demo_df.columns]
place_cols = [col for col in place_df.columns]
cost_cols = [col for col in cost_df.columns]
pin_cols = [col for col in pin_df.columns]

tower_dims = {
    'procedures': (0, 512),
    'diagnoses': (512, 1024),
    'demographics': (1024, 1029),
    'place': (1029, 1033),
    'cost': (1033, 1044),
    'pin': (1044, 1046)
}

print(f"✓ Data structures ready")
print(f"  Total providers: {len(all_pins_list)}")
print(f"  Procedure data: {len(pin_to_procedure_codes)} providers")
print(f"  Diagnosis data: {len(pin_to_diagnosis_codes)} providers")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cosine_similarity_manual(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b + 1e-8)

def compute_tower_similarity(emb_a, emb_b, start_idx, end_idx):
    tower_a = emb_a[start_idx:end_idx]
    tower_b = emb_b[start_idx:end_idx]
    return cosine_similarity_manual(tower_a, tower_b)

def compute_claims_overlap_metrics(pin_a_codes, pin_a_claims, pin_b_codes, pin_b_claims):
    """
    Compute claims-based overlap metrics.
    
    Returns:
        common_count: Number of overlapping codes
        a_claims_in_overlap: Total claims from A for overlapping codes
        b_claims_in_overlap: Total claims from B for overlapping codes
        a_pct_claims_overlap: % of A's total claims in overlap
        b_pct_claims_overlap: % of B's total claims in overlap
    """
    common_codes = pin_a_codes & pin_b_codes
    common_count = len(common_codes)
    
    if common_count == 0:
        return common_count, 0, 0, 0.0, 0.0
    
    a_claims_in_overlap = sum(pin_a_claims.get(code, 0) for code in common_codes)
    b_claims_in_overlap = sum(pin_b_claims.get(code, 0) for code in common_codes)
    
    a_total_claims = sum(pin_a_claims.values()) if pin_a_claims else 0
    b_total_claims = sum(pin_b_claims.values()) if pin_b_claims else 0
    
    a_pct_claims_overlap = (a_claims_in_overlap / a_total_claims * 100) if a_total_claims > 0 else 0.0
    b_pct_claims_overlap = (b_claims_in_overlap / b_total_claims * 100) if b_total_claims > 0 else 0.0
    
    return common_count, a_claims_in_overlap, b_claims_in_overlap, a_pct_claims_overlap, b_pct_claims_overlap

# ============================================================================
# PROCESS ALL PROVIDERS IN CHUNKS
# ============================================================================

print("\n" + "="*80)
print("GENERATING TOP 10 ALTERNATIVES FOR ALL PROVIDERS")
print("="*80)

CHUNK_SIZE = 500
TOP_K = 10

all_results = []
n_providers = len(all_pins_list)
n_chunks = (n_providers + CHUNK_SIZE - 1) // CHUNK_SIZE

print(f"Processing {n_providers:,} providers in {n_chunks} chunks")
print(f"Expected output rows: {n_providers * TOP_K:,}")

for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
    start_idx = chunk_idx * CHUNK_SIZE
    end_idx = min(start_idx + CHUNK_SIZE, n_providers)
    chunk_pins = all_pins_list[start_idx:end_idx]
    
    chunk_results = []
    
    for query_pin in chunk_pins:
        query_idx = pin_to_idx[query_pin]
        query_emb = pin_to_emb[query_pin]
        
        cnp_from_query = cnp_matrix[query_idx, :]
        cnp_from_query[query_idx] = 0
        
        top_k_indices = np.argsort(cnp_from_query)[-TOP_K:][::-1]
        
        for rank, rec_idx in enumerate(top_k_indices, 1):
            rec_pin = all_pins_list[rec_idx]
            rec_emb = pin_to_emb[rec_pin]
            
            cnp_a_to_b = cnp_from_query[rec_idx]
            cnp_b_to_a = cnp_matrix[rec_idx, query_idx]
            
            result = {
                'primary_pin': query_pin,
                'primary_name': pin_to_name.get(query_pin, 'Unknown'),
                'primary_specialty': pin_to_specialty.get(query_pin, 'Unknown'),
                'primary_label': pin_to_label.get(query_pin, 'UNLABELED'),
                
                'alternative_pin': rec_pin,
                'alternative_name': pin_to_name.get(rec_pin, 'Unknown'),
                'alternative_specialty': pin_to_specialty.get(rec_pin, 'Unknown'),
                'alternative_label': pin_to_label.get(rec_pin, 'UNLABELED'),
                
                'rank': rank,
                'cnp_primary_to_alternative': cnp_a_to_b,
                'cnp_alternative_to_primary': cnp_b_to_a
            }
            
            proc_codes_a = pin_to_procedure_codes.get(query_pin, set())
            proc_claims_a = pin_to_procedure_claims.get(query_pin, {})
            proc_codes_b = pin_to_procedure_codes.get(rec_pin, set())
            proc_claims_b = pin_to_procedure_claims.get(rec_pin, {})
            
            (proc_common_count, proc_a_claims_overlap, proc_b_claims_overlap,
             proc_a_pct_overlap, proc_b_pct_overlap) = compute_claims_overlap_metrics(
                proc_codes_a, proc_claims_a, proc_codes_b, proc_claims_b
            )
            
            result['primary_procedure_count'] = len(proc_codes_a)
            result['alternative_procedure_count'] = len(proc_codes_b)
            result['common_procedure_count'] = proc_common_count
            result['primary_procedure_claims_in_overlap'] = proc_a_claims_overlap
            result['alternative_procedure_claims_in_overlap'] = proc_b_claims_overlap
            result['primary_procedure_pct_claims_overlap'] = proc_a_pct_overlap
            result['alternative_procedure_pct_claims_overlap'] = proc_b_pct_overlap
            result['procedure_embedding_similarity'] = compute_tower_similarity(
                query_emb, rec_emb, tower_dims['procedures'][0], tower_dims['procedures'][1]
            )
            
            diag_codes_a = pin_to_diagnosis_codes.get(query_pin, set())
            diag_claims_a = pin_to_diagnosis_claims.get(query_pin, {})
            diag_codes_b = pin_to_diagnosis_codes.get(rec_pin, set())
            diag_claims_b = pin_to_diagnosis_claims.get(rec_pin, {})
            
            (diag_common_count, diag_a_claims_overlap, diag_b_claims_overlap,
             diag_a_pct_overlap, diag_b_pct_overlap) = compute_claims_overlap_metrics(
                diag_codes_a, diag_claims_a, diag_codes_b, diag_claims_b
            )
            
            result['primary_diagnosis_count'] = len(diag_codes_a)
            result['alternative_diagnosis_count'] = len(diag_codes_b)
            result['common_diagnosis_count'] = diag_common_count
            result['primary_diagnosis_claims_in_overlap'] = diag_a_claims_overlap
            result['alternative_diagnosis_claims_in_overlap'] = diag_b_claims_overlap
            result['primary_diagnosis_pct_claims_overlap'] = diag_a_pct_overlap
            result['alternative_diagnosis_pct_claims_overlap'] = diag_b_pct_overlap
            result['diagnosis_embedding_similarity'] = compute_tower_similarity(
                query_emb, rec_emb, tower_dims['diagnoses'][0], tower_dims['diagnoses'][1]
            )
            
            if query_pin in demo_df.index:
                for col in demo_cols:
                    result[f'primary_{col}'] = demo_df.loc[query_pin, col]
            else:
                for col in demo_cols:
                    result[f'primary_{col}'] = np.nan
            
            if rec_pin in demo_df.index:
                for col in demo_cols:
                    result[f'alternative_{col}'] = demo_df.loc[rec_pin, col]
            else:
                for col in demo_cols:
                    result[f'alternative_{col}'] = np.nan
            
            result['demographics_embedding_similarity'] = compute_tower_similarity(
                query_emb, rec_emb, tower_dims['demographics'][0], tower_dims['demographics'][1]
            )
            
            if query_pin in place_df.index:
                for col in place_cols:
                    result[f'primary_{col}'] = place_df.loc[query_pin, col]
            else:
                for col in place_cols:
                    result[f'primary_{col}'] = np.nan
            
            if rec_pin in place_df.index:
                for col in place_cols:
                    result[f'alternative_{col}'] = place_df.loc[rec_pin, col]
            else:
                for col in place_cols:
                    result[f'alternative_{col}'] = np.nan
            
            result['place_embedding_similarity'] = compute_tower_similarity(
                query_emb, rec_emb, tower_dims['place'][0], tower_dims['place'][1]
            )
            
            if query_pin in cost_df.index:
                for col in cost_cols:
                    result[f'primary_{col}'] = cost_df.loc[query_pin, col]
            else:
                for col in cost_cols:
                    result[f'primary_{col}'] = np.nan
            
            if rec_pin in cost_df.index:
                for col in cost_cols:
                    result[f'alternative_{col}'] = cost_df.loc[rec_pin, col]
            else:
                for col in cost_cols:
                    result[f'alternative_{col}'] = np.nan
            
            result['cost_embedding_similarity'] = compute_tower_similarity(
                query_emb, rec_emb, tower_dims['cost'][0], tower_dims['cost'][1]
            )
            
            if query_pin in pin_df.index:
                for col in pin_cols:
                    result[f'primary_{col}'] = pin_df.loc[query_pin, col]
            else:
                for col in pin_cols:
                    result[f'primary_{col}'] = np.nan
            
            if rec_pin in pin_df.index:
                for col in pin_cols:
                    result[f'alternative_{col}'] = pin_df.loc[rec_pin, col]
            else:
                for col in pin_cols:
                    result[f'alternative_{col}'] = np.nan
            
            result['pin_embedding_similarity'] = compute_tower_similarity(
                query_emb, rec_emb, tower_dims['pin'][0], tower_dims['pin'][1]
            )
            
            result['overall_embedding_similarity'] = cosine_similarity_manual(query_emb, rec_emb)
            
            chunk_results.append(result)
    
    chunk_df = pd.DataFrame(chunk_results)
    
    if chunk_idx == 0:
        chunk_df.to_csv('all_providers_top10_alternatives_me2vec.csv', index=False, mode='w')
    else:
        chunk_df.to_csv('all_providers_top10_alternatives_me2vec.csv', index=False, mode='a', header=False)
    
    all_results.extend(chunk_results)
    
    del chunk_results, chunk_df
    gc.collect()

print("\n✓ All providers processed!")
print(f"✓ Total rows generated: {len(all_results):,}")

# ============================================================================
# CREATE SPECIALTY CATEGORY DISTRIBUTION MATRIX
# ============================================================================

print("\n" + "="*80)
print("CREATING SPECIALTY CATEGORY DISTRIBUTION MATRIX")
print("="*80)

results_df = pd.DataFrame(all_results)

matrix_df = results_df[
    (results_df['primary_specialty'] != 'Unknown') & 
    (results_df['alternative_specialty'] != 'Unknown')
].copy()

print(f"Pairs with known specialties: {len(matrix_df):,}")

count_matrix = pd.crosstab(
    matrix_df['primary_specialty'],
    matrix_df['alternative_specialty'],
    margins=True,
    margins_name='Total'
)

print(f"\nSpecialty count matrix shape: {count_matrix.shape}")

pct_matrix = count_matrix.div(count_matrix['Total'], axis=0) * 100
pct_matrix = pct_matrix.round(2)

count_matrix.to_csv('specialty_category_count_matrix_me2vec.csv')
print("✓ Saved: specialty_category_count_matrix_me2vec.csv")

pct_matrix.to_csv('specialty_category_percentage_matrix_me2vec.csv')
print("✓ Saved: specialty_category_percentage_matrix_me2vec.csv")

combined_matrix = count_matrix.copy()
for col in combined_matrix.columns:
    if col != 'Total':
        combined_matrix[col] = combined_matrix[col].astype(str) + ' (' + pct_matrix[col].astype(str) + '%)'

combined_matrix.to_csv('specialty_category_combined_matrix_me2vec.csv')
print("✓ Saved: specialty_category_combined_matrix_me2vec.csv")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nTotal primary providers: {results_df['primary_pin'].nunique():,}")
print(f"Total alternative recommendations: {len(results_df):,}")

print(f"\nCNP Statistics:")
print(f"  CNP(Primary→Alternative) - used for ranking:")
print(f"    Mean: {results_df['cnp_primary_to_alternative'].mean():.6f}")
print(f"    Std:  {results_df['cnp_primary_to_alternative'].std():.6f}")
print(f"    Min:  {results_df['cnp_primary_to_alternative'].min():.6f}")
print(f"    Max:  {results_df['cnp_primary_to_alternative'].max():.6f}")
print(f"\n  CNP(Alternative→Primary) - for validation:")
print(f"    Mean: {results_df['cnp_alternative_to_primary'].mean():.6f}")
print(f"    Std:  {results_df['cnp_alternative_to_primary'].std():.6f}")

print(f"\nProcedure Claims Overlap:")
print(f"  Primary % claims in overlap:")
print(f"    Mean: {results_df['primary_procedure_pct_claims_overlap'].mean():.2f}%")
print(f"    Median: {results_df['primary_procedure_pct_claims_overlap'].median():.2f}%")
print(f"  Alternative % claims in overlap:")
print(f"    Mean: {results_df['alternative_procedure_pct_claims_overlap'].mean():.2f}%")
print(f"    Median: {results_df['alternative_procedure_pct_claims_overlap'].median():.2f}%")

print(f"\nDiagnosis Claims Overlap:")
print(f"  Primary % claims in overlap:")
print(f"    Mean: {results_df['primary_diagnosis_pct_claims_overlap'].mean():.2f}%")
print(f"    Median: {results_df['primary_diagnosis_pct_claims_overlap'].median():.2f}%")
print(f"  Alternative % claims in overlap:")
print(f"    Mean: {results_df['alternative_diagnosis_pct_claims_overlap'].mean():.2f}%")
print(f"    Median: {results_df['alternative_diagnosis_pct_claims_overlap'].median():.2f}%")

print(f"\nSame-specialty recommendations:")
same_specialty = (results_df['primary_specialty'] == results_df['alternative_specialty']).sum()
total_recs = len(results_df)
print(f"  Count: {same_specialty:,} ({same_specialty/total_recs:.1%})")

print("\n" + "="*80)
print("TOP 10 GENERATION COMPLETE - ME2VEC + CNP")
print("="*80)
print(f"\n✓ Generated top 10 alternatives for {results_df['primary_pin'].nunique():,} providers")
print(f"✓ Total recommendations: {len(results_df):,}")
print(f"\nOutput files:")
print(f"  1. all_providers_top10_alternatives_me2vec.csv")
print(f"  2. specialty_category_count_matrix_me2vec.csv")
print(f"  3. specialty_category_percentage_matrix_me2vec.csv")
print(f"  4. specialty_category_combined_matrix_me2vec.csv")
print(f"\nNew features:")
print(f"  ✓ CNP directional similarity (A→B and B→A)")
print(f"  ✓ Claims-based overlap metrics for procedures")
print(f"  ✓ Claims-based overlap metrics for diagnoses")
print(f"  ✓ % of claims in overlapping codes")
