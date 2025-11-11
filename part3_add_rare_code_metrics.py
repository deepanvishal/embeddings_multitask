import pandas as pd
import numpy as np
from tqdm import tqdm

print("\n" + "="*80)
print("PART 3: ADD RARE CODE METRICS TO TOP 10 ALTERNATIVES")
print("="*80)

RARITY_THRESHOLD = 0.70

print("\n" + "="*80)
print("STEP 1: LOAD DATA")
print("="*80)

print("\nLoading existing top 10 alternatives...")
top10_df = pd.read_csv('all_providers_top10_alternatives_me2vec.csv')
print(f"Top 10 alternatives shape: {top10_df.shape}")
print(f"Unique primary PINs: {top10_df['primary_pin'].nunique()}")

print("\nLoading rare codes...")
rare_codes_df = pd.read_parquet('rare_codes.parquet')
rare_codes_df = rare_codes_df[rare_codes_df['rarity_score'] >= RARITY_THRESHOLD]
rare_codes_set = set(rare_codes_df['code'].values)
print(f"Rare codes (rarity_score >= {RARITY_THRESHOLD}): {len(rare_codes_set)}")

print("\nLoading procedure data...")
procedure_df = pd.read_parquet('procedure_df.parquet')
print(f"Procedure data shape: {procedure_df.shape}")

print("\nLoading rare code embeddings...")
rare_embeddings_df = pd.read_csv('rare_codes_provider_embeddings.csv')
print(f"Rare code embeddings shape: {rare_embeddings_df.shape}")

emb_cols = [col for col in rare_embeddings_df.columns if col.startswith('emb_')]
pin_to_rare_emb = {}
for _, row in rare_embeddings_df.iterrows():
    pin = row['PIN']
    emb = row[emb_cols].values
    pin_to_rare_emb[pin] = emb

print(f"Providers with rare code embeddings: {len(pin_to_rare_emb)}")

print("\n" + "="*80)
print("STEP 2: PREPARE RARE CODE DATA STRUCTURES")
print("="*80)

print("\nFiltering procedure data to rare codes only...")
rare_procedure_df = procedure_df[procedure_df['code'].isin(rare_codes_set)].copy()
print(f"Rare procedure records: {len(rare_procedure_df)}")

print("\nBuilding PIN-to-rare-code mappings...")
rare_procedure_summary = rare_procedure_df.groupby('PIN').agg({
    'code': lambda x: set(x),
    'claims': lambda x: dict(zip(rare_procedure_df.loc[x.index, 'code'], x))
}).reset_index()
rare_procedure_summary.columns = ['PIN', 'rare_codes', 'rare_code_to_claims']

pin_to_rare_codes = dict(zip(rare_procedure_summary['PIN'], rare_procedure_summary['rare_codes']))
pin_to_rare_claims = dict(zip(rare_procedure_summary['PIN'], rare_procedure_summary['rare_code_to_claims']))

print(f"Providers with rare codes: {len(pin_to_rare_codes)}")

print("\n" + "="*80)
print("STEP 3: COMPUTE RARE CODE METRICS")
print("="*80)

def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b + 1e-8)

def compute_rare_code_overlap(pin_a, pin_b):
    rare_codes_a = pin_to_rare_codes.get(pin_a, set())
    rare_codes_b = pin_to_rare_codes.get(pin_b, set())
    rare_claims_a = pin_to_rare_claims.get(pin_a, {})
    rare_claims_b = pin_to_rare_claims.get(pin_b, {})
    
    common_rare_codes = rare_codes_a & rare_codes_b
    common_count = len(common_rare_codes)
    
    if common_count == 0:
        return {
            'primary_rare_code_count': len(rare_codes_a),
            'alternative_rare_code_count': len(rare_codes_b),
            'common_rare_code_count': 0,
            'primary_rare_claims_in_overlap': 0,
            'alternative_rare_claims_in_overlap': 0,
            'primary_rare_pct_claims_overlap': 0.0,
            'alternative_rare_pct_claims_overlap': 0.0
        }
    
    a_claims_in_overlap = sum(rare_claims_a.get(code, 0) for code in common_rare_codes)
    b_claims_in_overlap = sum(rare_claims_b.get(code, 0) for code in common_rare_codes)
    
    a_total_claims = sum(rare_claims_a.values()) if rare_claims_a else 0
    b_total_claims = sum(rare_claims_b.values()) if rare_claims_b else 0
    
    a_pct_claims_overlap = (a_claims_in_overlap / a_total_claims * 100) if a_total_claims > 0 else 0.0
    b_pct_claims_overlap = (b_claims_in_overlap / b_total_claims * 100) if b_total_claims > 0 else 0.0
    
    return {
        'primary_rare_code_count': len(rare_codes_a),
        'alternative_rare_code_count': len(rare_codes_b),
        'common_rare_code_count': common_count,
        'primary_rare_claims_in_overlap': a_claims_in_overlap,
        'alternative_rare_claims_in_overlap': b_claims_in_overlap,
        'primary_rare_pct_claims_overlap': a_pct_claims_overlap,
        'alternative_rare_pct_claims_overlap': b_pct_claims_overlap
    }

def compute_rare_embedding_similarity(pin_a, pin_b):
    emb_a = pin_to_rare_emb.get(pin_a)
    emb_b = pin_to_rare_emb.get(pin_b)
    
    if emb_a is None or emb_b is None:
        return np.nan
    
    return cosine_similarity(emb_a, emb_b)

print("\nComputing rare code metrics for all pairs...")
rare_metrics_list = []

for idx, row in tqdm(top10_df.iterrows(), total=len(top10_df), desc="Processing pairs"):
    primary_pin = row['primary_pin']
    alternative_pin = row['alternative_pin']
    
    overlap_metrics = compute_rare_code_overlap(primary_pin, alternative_pin)
    rare_emb_sim = compute_rare_embedding_similarity(primary_pin, alternative_pin)
    
    overlap_metrics['rare_code_embedding_similarity'] = rare_emb_sim
    
    rare_metrics_list.append(overlap_metrics)

rare_metrics_df = pd.DataFrame(rare_metrics_list)

print(f"\nRare metrics computed for {len(rare_metrics_df)} pairs")

print("\n" + "="*80)
print("STEP 4: MERGE AND SAVE RESULTS")
print("="*80)

result_df = pd.concat([top10_df, rare_metrics_df], axis=1)

print(f"\nFinal dataframe shape: {result_df.shape}")
print(f"New columns added: {list(rare_metrics_df.columns)}")

output_file = 'all_providers_top10_alternatives_me2vec_with_rare_codes.csv'
result_df.to_csv(output_file, index=False)
print(f"\nSaved: {output_file}")

print("\n" + "="*80)
print("STEP 5: SUMMARY STATISTICS")
print("="*80)

print(f"\nRare Code Counts:")
print(f"  Primary rare code count:")
print(f"    Mean: {result_df['primary_rare_code_count'].mean():.2f}")
print(f"    Median: {result_df['primary_rare_code_count'].median():.2f}")
print(f"    Max: {result_df['primary_rare_code_count'].max():.0f}")

print(f"\n  Alternative rare code count:")
print(f"    Mean: {result_df['alternative_rare_code_count'].mean():.2f}")
print(f"    Median: {result_df['alternative_rare_code_count'].median():.2f}")
print(f"    Max: {result_df['alternative_rare_code_count'].max():.0f}")

print(f"\nRare Code Overlap:")
print(f"  Common rare codes:")
print(f"    Mean: {result_df['common_rare_code_count'].mean():.2f}")
print(f"    Median: {result_df['common_rare_code_count'].median():.2f}")
print(f"    Max: {result_df['common_rare_code_count'].max():.0f}")

print(f"\n  Pairs with rare code overlap: {(result_df['common_rare_code_count'] > 0).sum():,} ({(result_df['common_rare_code_count'] > 0).mean():.1%})")
print(f"  Pairs with no rare code overlap: {(result_df['common_rare_code_count'] == 0).sum():,} ({(result_df['common_rare_code_count'] == 0).mean():.1%})")

print(f"\nRare Claims Overlap:")
print(f"  Primary % rare claims in overlap:")
print(f"    Mean: {result_df['primary_rare_pct_claims_overlap'].mean():.2f}%")
print(f"    Median: {result_df['primary_rare_pct_claims_overlap'].median():.2f}%")

print(f"\n  Alternative % rare claims in overlap:")
print(f"    Mean: {result_df['alternative_rare_pct_claims_overlap'].mean():.2f}%")
print(f"    Median: {result_df['alternative_rare_pct_claims_overlap'].median():.2f}%")

print(f"\nRare Code Embedding Similarity:")
valid_sims = result_df['rare_code_embedding_similarity'].dropna()
print(f"  Valid similarities: {len(valid_sims):,} ({len(valid_sims)/len(result_df):.1%})")
print(f"  Mean: {valid_sims.mean():.6f}")
print(f"  Median: {valid_sims.median():.6f}")
print(f"  Std: {valid_sims.std():.6f}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nOutput: {output_file}")
print(f"Shape: {result_df.shape}")
print(f"\nNew columns added:")
for col in rare_metrics_df.columns:
    print(f"  - {col}")

print("\nSample of new columns:")
sample_cols = ['primary_pin', 'alternative_pin', 'primary_rare_code_count', 
               'alternative_rare_code_count', 'common_rare_code_count', 
               'rare_code_embedding_similarity']
print(result_df[sample_cols].head(10))
