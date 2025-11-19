"""
THRESHOLD APPLICATION & DISTRIBUTION ANALYSIS
=============================================

Applies a similarity threshold to find good/bad pairs and analyzes:
1. How many providers have N good alternatives
2. Distribution bucketed by ranges (0, 1-5, 5-10, etc.)

Uses vectorized operations for speed.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
SIMILARITY_THRESHOLD = 0.75  # CHANGE THIS VALUE TO TEST DIFFERENT THRESHOLDS
# =============================================================================

print("\n" + "="*80)
print("THRESHOLD APPLICATION & DISTRIBUTION ANALYSIS")
print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
print("="*80)

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

embeddings_df = pd.read_parquet('final_me2vec_all_towers_1046d.parquet')
print(f"Embeddings: {embeddings_df.shape}")

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)
print(f"Labels: {len(pin_to_label)}")

prov_spl_df = pd.read_parquet('prov_spl.parquet')
pin_to_specialty = dict(zip(prov_spl_df['PIN'], prov_spl_df['srv_spclty_ctg_cd']))
print(f"Specialties: {len(pin_to_specialty)}")

county_df = pd.read_parquet('county_df.parquet')
print(f"County data: {county_df.shape}")

county_df['county_state'] = county_df['county_nm'].fillna('') + '|' + county_df['state_postal_cd'].fillna('')
pin_to_county_state = dict(zip(county_df['PIN'], county_df['county_state']))
print(f"PIN to county mapping: {len(pin_to_county_state)}")

county_state_to_pins = {}
for pin, county_state in pin_to_county_state.items():
    if county_state not in county_state_to_pins:
        county_state_to_pins[county_state] = []
    county_state_to_pins[county_state].append(pin)

print(f"\nCounty distribution:")
print(f"  Unique county+state combinations: {len(county_state_to_pins)}")
print(f"  Providers with blank county: {len(county_state_to_pins.get('|', []))}")

all_pins = embeddings_df['PIN'].values
n_providers = len(all_pins)
print(f"Total providers: {n_providers:,}")

emb_cols = [col for col in embeddings_df.columns if col != 'PIN']
embeddings_df = embeddings_df.set_index('PIN')
embeddings_matrix = embeddings_df[emb_cols].values
print(f"Embedding dimension: {embeddings_matrix.shape[1]}")

print("\n" + "="*80)
print("COMPUTING ALL SAME-COUNTY PAIRS (VECTORIZED)")
print("="*80)

pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

print("\nBuilding county-to-index mappings...")
county_to_indices = {}
for county_state, pins in county_state_to_pins.items():
    if county_state == '|':
        continue
    indices = [pin_to_idx[pin] for pin in pins if pin in pin_to_idx]
    if len(indices) >= 2:
        county_to_indices[county_state] = indices

eligible_counties = list(county_to_indices.keys())
print(f"Eligible counties (≥2 providers): {len(eligible_counties)}")

total_pairs = sum(len(indices) * (len(indices) - 1) // 2 for indices in county_to_indices.values())
print(f"Total possible same-county pairs: {total_pairs:,}")

print("\nComputing similarities county by county (vectorized)...")

all_pairs_data = []

for county_idx, (county_state, indices) in enumerate(tqdm(county_to_indices.items(), desc="Processing counties")):
    n = len(indices)
    
    if n < 2:
        continue
    
    county_embeddings = embeddings_matrix[indices]
    
    # Vectorized cosine similarity computation
    # Normalize embeddings
    norms = np.linalg.norm(county_embeddings, axis=1, keepdims=True)
    normalized = county_embeddings / (norms + 1e-8)
    
    # Compute similarity matrix: dot product of normalized vectors
    sim_matrix = np.dot(normalized, normalized.T)
    
    # Extract upper triangle (exclude diagonal and lower triangle)
    upper_tri_indices = np.triu_indices(n, k=1)
    
    for i, j in zip(upper_tri_indices[0], upper_tri_indices[1]):
        idx_a = indices[i]
        idx_b = indices[j]
        similarity = sim_matrix[i, j]
        
        pin_a = all_pins[idx_a]
        pin_b = all_pins[idx_b]
        
        all_pairs_data.append({
            'pin_a': pin_a,
            'pin_b': pin_b,
            'idx_a': idx_a,
            'idx_b': idx_b,
            'similarity': similarity,
            'county_state': county_state,
            'label_a': pin_to_label.get(pin_a, 'Unknown'),
            'label_b': pin_to_label.get(pin_b, 'Unknown'),
            'specialty_a': pin_to_specialty.get(pin_a, 'Unknown'),
            'specialty_b': pin_to_specialty.get(pin_b, 'Unknown')
        })

print(f"\n✓ Computed {len(all_pairs_data):,} same-county pairs")

pairs_df = pd.DataFrame(all_pairs_data)

print("\n" + "="*80)
print("APPLYING THRESHOLD")
print("="*80)

pairs_df['is_good_pair'] = pairs_df['similarity'] >= SIMILARITY_THRESHOLD
pairs_df['is_bad_pair'] = pairs_df['similarity'] < SIMILARITY_THRESHOLD

n_good = pairs_df['is_good_pair'].sum()
n_bad = pairs_df['is_bad_pair'].sum()

print(f"\nThreshold: {SIMILARITY_THRESHOLD}")
print(f"  Good pairs (≥ threshold): {n_good:,} ({n_good/len(pairs_df)*100:.1f}%)")
print(f"  Bad pairs (< threshold): {n_bad:,} ({n_bad/len(pairs_df)*100:.1f}%)")

print(f"\nSimilarity statistics:")
print(f"  Min:    {pairs_df['similarity'].min():.6f}")
print(f"  Q1:     {pairs_df['similarity'].quantile(0.25):.6f}")
print(f"  Median: {pairs_df['similarity'].median():.6f}")
print(f"  Q3:     {pairs_df['similarity'].quantile(0.75):.6f}")
print(f"  Max:    {pairs_df['similarity'].max():.6f}")
print(f"  Mean:   {pairs_df['similarity'].mean():.6f}")

print("\n" + "="*80)
print("PER-PROVIDER ANALYSIS")
print("="*80)

good_pairs_df = pairs_df[pairs_df['is_good_pair']].copy()

print("\nCounting good alternatives per provider...")

provider_good_count_a = good_pairs_df.groupby('pin_a').size()
provider_good_count_b = good_pairs_df.groupby('pin_b').size()

all_provider_pins = set(pairs_df['pin_a'].unique()) | set(pairs_df['pin_b'].unique())

provider_alternatives = {}
for pin in all_provider_pins:
    count_a = provider_good_count_a.get(pin, 0)
    count_b = provider_good_count_b.get(pin, 0)
    provider_alternatives[pin] = count_a + count_b

provider_alt_df = pd.DataFrame([
    {'PIN': pin, 'n_good_alternatives': count}
    for pin, count in provider_alternatives.items()
])

provider_alt_df['label'] = provider_alt_df['PIN'].map(pin_to_label).fillna('Unknown')
provider_alt_df['specialty'] = provider_alt_df['PIN'].map(pin_to_specialty).fillna('Unknown')
provider_alt_df['county_state'] = provider_alt_df['PIN'].map(pin_to_county_state).fillna('|')

print(f"\n✓ Analyzed {len(provider_alt_df)} providers")

print("\n" + "="*80)
print("DISTRIBUTION ANALYSIS - EXACT COUNTS")
print("="*80)

exact_counts = provider_alt_df['n_good_alternatives'].value_counts().sort_index()

print(f"\nProviders by exact number of good alternatives:")
print(f"{'N Alternatives':<15} {'N Providers':<15} {'%':<10}")
print("-" * 40)

for n_alt, n_prov in exact_counts.items():
    pct = n_prov / len(provider_alt_df) * 100
    print(f"{n_alt:<15} {n_prov:<15} {pct:>6.2f}%")

print(f"\nTop 20 most common counts:")
top_20 = exact_counts.head(20)
for n_alt, n_prov in top_20.items():
    pct = n_prov / len(provider_alt_df) * 100
    print(f"  {n_alt} alternatives: {n_prov} providers ({pct:.2f}%)")

print("\n" + "="*80)
print("DISTRIBUTION ANALYSIS - BUCKETED")
print("="*80)

def bucket_alternatives(n):
    if n == 0:
        return '0'
    elif n <= 5:
        return '1-5'
    elif n <= 10:
        return '6-10'
    elif n <= 20:
        return '11-20'
    elif n <= 30:
        return '21-30'
    elif n <= 40:
        return '31-40'
    elif n <= 50:
        return '41-50'
    else:
        return '50+'

provider_alt_df['bucket'] = provider_alt_df['n_good_alternatives'].apply(bucket_alternatives)

bucket_counts = provider_alt_df['bucket'].value_counts()

bucket_order = ['0', '1-5', '6-10', '11-20', '21-30', '31-40', '41-50', '50+']
bucket_counts = bucket_counts.reindex(bucket_order, fill_value=0)

print(f"\nProviders by alternative count buckets:")
print(f"{'Bucket':<15} {'N Providers':<15} {'%':<10}")
print("-" * 40)

for bucket, count in bucket_counts.items():
    pct = count / len(provider_alt_df) * 100
    print(f"{bucket:<15} {count:<15} {pct:>6.2f}%")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nOverall statistics:")
print(f"  Total providers analyzed: {len(provider_alt_df):,}")
print(f"  Mean alternatives per provider: {provider_alt_df['n_good_alternatives'].mean():.2f}")
print(f"  Median alternatives per provider: {provider_alt_df['n_good_alternatives'].median():.0f}")
print(f"  Std dev: {provider_alt_df['n_good_alternatives'].std():.2f}")
print(f"  Min alternatives: {provider_alt_df['n_good_alternatives'].min()}")
print(f"  Max alternatives: {provider_alt_df['n_good_alternatives'].max()}")

providers_with_zero = (provider_alt_df['n_good_alternatives'] == 0).sum()
providers_with_any = (provider_alt_df['n_good_alternatives'] > 0).sum()

print(f"\nCoverage:")
print(f"  Providers with 0 alternatives: {providers_with_zero:,} ({providers_with_zero/len(provider_alt_df)*100:.1f}%)")
print(f"  Providers with ≥1 alternatives: {providers_with_any:,} ({providers_with_any/len(provider_alt_df)*100:.1f}%)")
print(f"  Providers with ≥5 alternatives: {(provider_alt_df['n_good_alternatives'] >= 5).sum():,}")
print(f"  Providers with ≥10 alternatives: {(provider_alt_df['n_good_alternatives'] >= 10).sum():,}")

print("\n" + "="*80)
print("VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Similarity distribution with threshold
axes[0, 0].hist(pairs_df['similarity'], bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(SIMILARITY_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {SIMILARITY_THRESHOLD}')
axes[0, 0].set_xlabel('Similarity')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Similarity Distribution (All Pairs)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Good vs bad pairs
axes[0, 1].bar(['Good Pairs', 'Bad Pairs'], [n_good, n_bad], color=['green', 'red'], alpha=0.7)
axes[0, 1].set_ylabel('Number of Pairs')
axes[0, 1].set_title(f'Pairs by Threshold ({SIMILARITY_THRESHOLD})')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, (label, val) in enumerate(zip(['Good Pairs', 'Bad Pairs'], [n_good, n_bad])):
    axes[0, 1].text(i, val, f'{val:,}\n({val/len(pairs_df)*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')

# 3. Alternatives per provider distribution
axes[0, 2].hist(provider_alt_df['n_good_alternatives'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].axvline(provider_alt_df['n_good_alternatives'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {provider_alt_df["n_good_alternatives"].mean():.1f}')
axes[0, 2].axvline(provider_alt_df['n_good_alternatives'].median(), color='orange', 
                   linestyle='--', linewidth=2, label=f'Median: {provider_alt_df["n_good_alternatives"].median():.0f}')
axes[0, 2].set_xlabel('Number of Good Alternatives')
axes[0, 2].set_ylabel('Number of Providers')
axes[0, 2].set_title('Distribution: Alternatives per Provider')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Bucketed distribution
bucket_counts.plot(kind='bar', ax=axes[1, 0], color='steelblue', alpha=0.7)
axes[1, 0].set_xlabel('Alternative Count Bucket')
axes[1, 0].set_ylabel('Number of Providers')
axes[1, 0].set_title('Bucketed Distribution')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, (bucket, val) in enumerate(bucket_counts.items()):
    axes[1, 0].text(i, val, f'{val:,}', ha='center', va='bottom', fontsize=9)

# 5. Cumulative distribution
sorted_alts = np.sort(provider_alt_df['n_good_alternatives'].values)
cumulative = np.arange(1, len(sorted_alts) + 1) / len(sorted_alts) * 100
axes[1, 1].plot(sorted_alts, cumulative, linewidth=2)
axes[1, 1].set_xlabel('Number of Good Alternatives')
axes[1, 1].set_ylabel('Cumulative % of Providers')
axes[1, 1].set_title('Cumulative Distribution')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
axes[1, 1].axhline(90, color='orange', linestyle='--', alpha=0.5, label='90%')
axes[1, 1].legend()

# 6. Top 30 exact counts
top_30 = exact_counts.head(30)
axes[1, 2].bar(range(len(top_30)), top_30.values, color='purple', alpha=0.7)
axes[1, 2].set_xlabel('Number of Alternatives')
axes[1, 2].set_ylabel('Number of Providers')
axes[1, 2].set_title('Top 30 Most Common Alternative Counts')
axes[1, 2].set_xticks(range(len(top_30)))
axes[1, 2].set_xticklabels(top_30.index, rotation=45, ha='right')
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'distribution_analysis_threshold_{SIMILARITY_THRESHOLD}.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: distribution_analysis_threshold_{SIMILARITY_THRESHOLD}.png")

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

pairs_df.to_parquet(f'all_same_county_pairs_threshold_{SIMILARITY_THRESHOLD}.parquet')
print(f"✓ Saved: all_same_county_pairs_threshold_{SIMILARITY_THRESHOLD}.parquet")

provider_alt_df.to_csv(f'provider_alternatives_count_threshold_{SIMILARITY_THRESHOLD}.csv', index=False)
print(f"✓ Saved: provider_alternatives_count_threshold_{SIMILARITY_THRESHOLD}.csv")

exact_counts.to_csv(f'exact_count_distribution_threshold_{SIMILARITY_THRESHOLD}.csv', header=['n_providers'])
print(f"✓ Saved: exact_count_distribution_threshold_{SIMILARITY_THRESHOLD}.csv")

bucket_counts.to_csv(f'bucketed_distribution_threshold_{SIMILARITY_THRESHOLD}.csv', header=['n_providers'])
print(f"✓ Saved: bucketed_distribution_threshold_{SIMILARITY_THRESHOLD}.csv")

summary_stats = {
    'threshold': SIMILARITY_THRESHOLD,
    'total_pairs': len(pairs_df),
    'good_pairs': n_good,
    'bad_pairs': n_bad,
    'total_providers': len(provider_alt_df),
    'providers_with_zero_alts': providers_with_zero,
    'providers_with_any_alts': providers_with_any,
    'mean_alts_per_provider': provider_alt_df['n_good_alternatives'].mean(),
    'median_alts_per_provider': provider_alt_df['n_good_alternatives'].median(),
    'min_alts': provider_alt_df['n_good_alternatives'].min(),
    'max_alts': provider_alt_df['n_good_alternatives'].max()
}

with open(f'summary_stats_threshold_{SIMILARITY_THRESHOLD}.pkl', 'wb') as f:
    pickle.dump(summary_stats, f)
print(f"✓ Saved: summary_stats_threshold_{SIMILARITY_THRESHOLD}.pkl")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nTo test different thresholds:")
print(f"  1. Change SIMILARITY_THRESHOLD at top of file (line 23)")
print(f"  2. Re-run the script")
print(f"\nCurrent threshold: {SIMILARITY_THRESHOLD}")
print(f"Files saved with threshold suffix: _threshold_{SIMILARITY_THRESHOLD}")
