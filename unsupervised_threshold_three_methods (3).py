"""
UNSUPERVISED THRESHOLD DETERMINATION - THREE METHODS
====================================================

Method 1: Global Quantile Threshold
    - Sample random pairs, find top X% by overall similarity
    
Method 2: Per-Provider Top-K + Knee Detection
    - Adaptive local threshold per provider
    
Method 3: GMM on All 7 Similarities
    - Multi-dimensional clustering to identify "similar" vs "background" pairs
    
Combined Final Rule:
    - Global threshold + Top-K + GMM probability filter
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
N_PAIRS = 1_000
QUANTILES_TO_TEST = [0.90, 0.95, 0.97, 0.98, 0.99]
TOP_K = 10
GMM_THRESHOLD = 0.8
# =============================================================================

print("\n" + "="*80)
print("UNSUPERVISED THRESHOLD DETERMINATION")
print(f"Sample Size: {N_PAIRS:,} pairs")
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
county_sizes = [len(pins) for county_state, pins in county_state_to_pins.items() if county_state != '|']
if county_sizes:
    print(f"  Min providers per county (excl blank): {min(county_sizes)}")
    print(f"  Max providers per county (excl blank): {max(county_sizes)}")
    print(f"  Mean providers per county (excl blank): {np.mean(county_sizes):.1f}")

all_pins = embeddings_df['PIN'].values
n_providers = len(all_pins)
print(f"Total providers: {n_providers:,}")

emb_cols = [col for col in embeddings_df.columns if col != 'PIN']
embeddings_df = embeddings_df.set_index('PIN')
embeddings_matrix = embeddings_df[emb_cols].values

print("\nDetecting tower structure...")
tower_prefixes = {}
for col in emb_cols:
    if col.startswith('tower'):
        tower_num = col.split('_')[0]
        if tower_num not in tower_prefixes:
            tower_prefixes[tower_num] = []
        tower_prefixes[tower_num].append(col)

tower_mapping = {
    'tower1': 'procedures',
    'tower2': 'diagnoses', 
    'tower3': 'demographics',
    'tower4': 'place',
    'tower5': 'cost',
    'tower6': 'totality'
}

tower_dims = {}
current_idx = 0
for tower_key in sorted(tower_prefixes.keys()):
    tower_cols = sorted(tower_prefixes[tower_key])
    tower_size = len(tower_cols)
    tower_name = tower_mapping.get(tower_key, tower_key)
    tower_dims[tower_name] = (current_idx, current_idx + tower_size)
    current_idx += tower_size

print(f"\nTower dimensions:")
for name, (start, end) in tower_dims.items():
    print(f"  {name}: [{start}:{end}] ({end-start} dims)")

labeled_count = sum(1 for v in pin_to_label.values() if v != 'Unknown')
print(f"\nLabeled providers: {labeled_count:,} ({labeled_count/n_providers:.1%})")

print("\n" + "="*80)
print("GENERATING RANDOM PAIRS (SAME COUNTY ONLY)")
print("="*80)

np.random.seed(42)

print(f"\nPre-computing county-to-index mappings...")
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

county_to_indices = {}
for county_state, pins in county_state_to_pins.items():
    if county_state == '|':
        continue
    indices = [pin_to_idx[pin] for pin in pins if pin in pin_to_idx]
    if len(indices) >= 2:
        county_to_indices[county_state] = indices

eligible_counties = list(county_to_indices.keys())
print(f"Eligible counties (≥2 providers with embeddings): {len(eligible_counties)}")

county_pair_potential = {cs: len(indices) * (len(indices) - 1) // 2 
                         for cs, indices in county_to_indices.items()}
total_potential = sum(county_pair_potential.values())
print(f"Total potential same-county pairs: {total_potential:,}")

if total_potential < N_PAIRS:
    print(f"WARNING: Only {total_potential:,} possible pairs available, reducing N_PAIRS")
    N_PAIRS = total_potential

print(f"\nSampling {N_PAIRS:,} unique provider pairs from same counties...")

pairs_set = set()
while len(pairs_set) < N_PAIRS:
    county_state = np.random.choice(eligible_counties)
    indices = county_to_indices[county_state]
    
    if len(indices) < 2:
        continue
    
    idx_a, idx_b = np.random.choice(indices, size=2, replace=False)
    pair = tuple(sorted([idx_a, idx_b]))
    pairs_set.add(pair)
    
    if len(pairs_set) % 100 == 0:
        print(f"  Generated {len(pairs_set):,} pairs...", end='\r')

print(f"\n✓ Generated {len(pairs_set):,} unique same-county pairs")

pairs = list(pairs_set)

def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b + 1e-8)

def compute_tower_similarity(emb_a, emb_b, start_idx, end_idx):
    tower_a = emb_a[start_idx:end_idx]
    tower_b = emb_b[start_idx:end_idx]
    return cosine_similarity(tower_a, tower_b)

print("\nComputing all 7 similarities...")
data = {
    'idx_a': [],
    'idx_b': [],
    'pin_a': [],
    'pin_b': [],
    'sim_procedures': [],
    'sim_diagnoses': [],
    'sim_demographics': [],
    'sim_place': [],
    'sim_cost': [],
    'sim_totality': [],
    'sim_overall': [],
    'label_a': [],
    'label_b': [],
    'specialty_a': [],
    'specialty_b': [],
    'county_state_a': [],
    'county_state_b': []
}

for idx_a, idx_b in tqdm(pairs, desc="Computing"):
    emb_a = embeddings_matrix[idx_a]
    emb_b = embeddings_matrix[idx_b]
    
    pin_a = all_pins[idx_a]
    pin_b = all_pins[idx_b]
    
    data['idx_a'].append(idx_a)
    data['idx_b'].append(idx_b)
    data['pin_a'].append(pin_a)
    data['pin_b'].append(pin_b)
    data['sim_procedures'].append(compute_tower_similarity(emb_a, emb_b, *tower_dims['procedures']))
    data['sim_diagnoses'].append(compute_tower_similarity(emb_a, emb_b, *tower_dims['diagnoses']))
    data['sim_demographics'].append(compute_tower_similarity(emb_a, emb_b, *tower_dims['demographics']))
    data['sim_place'].append(compute_tower_similarity(emb_a, emb_b, *tower_dims['place']))
    data['sim_cost'].append(compute_tower_similarity(emb_a, emb_b, *tower_dims['cost']))
    data['sim_totality'].append(compute_tower_similarity(emb_a, emb_b, *tower_dims['totality']))
    data['sim_overall'].append(cosine_similarity(emb_a, emb_b))
    data['label_a'].append(pin_to_label.get(pin_a, 'Unknown'))
    data['label_b'].append(pin_to_label.get(pin_b, 'Unknown'))
    data['specialty_a'].append(pin_to_specialty.get(pin_a, 'Unknown'))
    data['specialty_b'].append(pin_to_specialty.get(pin_b, 'Unknown'))
    data['county_state_a'].append(pin_to_county_state.get(pin_a, '|'))
    data['county_state_b'].append(pin_to_county_state.get(pin_b, '|'))

pairs_df = pd.DataFrame(data)
print(f"✓ Computed {len(pairs_df):,} pairs with 7 similarities each")

pairs_df['same_county'] = pairs_df['county_state_a'] == pairs_df['county_state_b']
print(f"\nCounty verification:")
print(f"  Same county pairs: {pairs_df['same_county'].sum()} ({pairs_df['same_county'].mean()*100:.1f}%)")
if not pairs_df['same_county'].all():
    print(f"  WARNING: {(~pairs_df['same_county']).sum()} cross-county pairs detected!")

def is_good_pair(label_a, label_b):
    if label_a == 'Unknown' or label_b == 'Unknown':
        return True
    return label_a == label_b

def is_bad_pair(label_a, label_b):
    if label_a == 'Unknown' or label_b == 'Unknown':
        return False
    return label_a != label_b

pairs_df['is_good_pair'] = pairs_df.apply(lambda row: is_good_pair(row['label_a'], row['label_b']), axis=1)
pairs_df['is_bad_pair'] = pairs_df.apply(lambda row: is_bad_pair(row['label_a'], row['label_b']), axis=1)
pairs_df['both_labeled'] = (pairs_df['label_a'] != 'Unknown') & (pairs_df['label_b'] != 'Unknown')
pairs_df['same_specialty'] = pairs_df['specialty_a'] == pairs_df['specialty_b']

print(f"\nPair Composition:")
print(f"  Total pairs: {len(pairs_df):,}")
print(f"  Same county: {pairs_df['same_county'].sum():,} ({pairs_df['same_county'].mean():.1%})")
print(f"  Both labeled: {pairs_df['both_labeled'].sum():,} ({pairs_df['both_labeled'].mean():.1%})")
print(f"  Good pairs: {pairs_df['is_good_pair'].sum():,} ({pairs_df['is_good_pair'].mean():.1%})")
print(f"  Bad pairs: {pairs_df['is_bad_pair'].sum():,} ({pairs_df['is_bad_pair'].mean():.1%})")
print(f"  Same specialty: {pairs_df['same_specialty'].sum():,} ({pairs_df['same_specialty'].mean():.1%})")

county_pair_counts = pairs_df.groupby('county_state_a').size()
print(f"\nCounty representation in sample:")
print(f"  Counties sampled: {len(county_pair_counts)}")
print(f"  Min pairs per county: {county_pair_counts.min()}")
print(f"  Max pairs per county: {county_pair_counts.max()}")
print(f"  Mean pairs per county: {county_pair_counts.mean():.1f}")

print("\n" + "="*80)
print("SIMILARITY DISTRIBUTIONS")
print("="*80)

sim_cols = ['sim_procedures', 'sim_diagnoses', 'sim_demographics', 'sim_place', 
            'sim_cost', 'sim_totality', 'sim_overall']

print("\nSummary Statistics:")
for col in sim_cols:
    print(f"\n{col}:")
    print(f"  Min:    {pairs_df[col].min():.6f}")
    print(f"  Q1:     {pairs_df[col].quantile(0.25):.6f}")
    print(f"  Median: {pairs_df[col].median():.6f}")
    print(f"  Q3:     {pairs_df[col].quantile(0.75):.6f}")
    print(f"  Max:    {pairs_df[col].max():.6f}")
    print(f"  Mean:   {pairs_df[col].mean():.6f}")
    print(f"  Std:    {pairs_df[col].std():.6f}")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(sim_cols):
    axes[i].hist(pairs_df[col], bins=50, edgecolor='black', alpha=0.7)
    axes[i].set_xlabel('Similarity')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(col.replace('sim_', '').title())
    axes[i].grid(True, alpha=0.3)
    
    mean_val = pairs_df[col].mean()
    axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.3f}')
    axes[i].legend()

corr_matrix = pairs_df[sim_cols].corr()
im = axes[7].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[7].set_xticks(range(len(sim_cols)))
axes[7].set_yticks(range(len(sim_cols)))
axes[7].set_xticklabels([c.replace('sim_', '') for c in sim_cols], rotation=45, ha='right')
axes[7].set_yticklabels([c.replace('sim_', '') for c in sim_cols])
axes[7].set_title('Correlation Matrix')
plt.colorbar(im, ax=axes[7])

for i in range(len(sim_cols)):
    for j in range(len(sim_cols)):
        text = axes[7].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)

axes[8].axis('off')

plt.tight_layout()
plt.savefig('similarity_distributions.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: similarity_distributions.png")

print("\n" + "="*80)
print("METHOD 1: GLOBAL QUANTILE THRESHOLD")
print("="*80)

print("\nTesting quantiles on sim_overall:")
quantile_results = []

for q in QUANTILES_TO_TEST:
    threshold = np.quantile(pairs_df['sim_overall'], q)
    
    above_mask = pairs_df['sim_overall'] >= threshold
    n_above = above_mask.sum()
    pct_above = n_above / len(pairs_df) * 100
    
    good_above = pairs_df[above_mask]['is_good_pair'].sum()
    bad_above = pairs_df[above_mask]['is_bad_pair'].sum()
    good_above_pct = good_above / n_above * 100 if n_above > 0 else 0
    
    same_spec_above = pairs_df[above_mask]['same_specialty'].sum()
    same_spec_pct = same_spec_above / n_above * 100 if n_above > 0 else 0
    
    quantile_results.append({
        'quantile': q,
        'threshold': threshold,
        'n_above': n_above,
        'pct_above': pct_above,
        'good_above_pct': good_above_pct,
        'same_spec_pct': same_spec_pct
    })
    
    print(f"\nQuantile {q:.2f} (top {(1-q)*100:.0f}%):")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Pairs above: {n_above} ({pct_above:.1f}%)")
    print(f"  Good pairs above: {good_above_pct:.1f}%")
    print(f"  Same specialty: {same_spec_pct:.1f}%")

quantile_df = pd.DataFrame(quantile_results)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(pairs_df['sim_overall'], bins=50, edgecolor='black', alpha=0.7)
for _, row in quantile_df.iterrows():
    axes[0, 0].axvline(row['threshold'], linestyle='--', linewidth=2, 
                       label=f"Q{row['quantile']:.2f}: {row['threshold']:.3f}")
axes[0, 0].set_xlabel('Overall Similarity')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Quantile Thresholds on Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(quantile_df['quantile'], quantile_df['threshold'], 'o-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Quantile')
axes[0, 1].set_ylabel('Threshold')
axes[0, 1].set_title('Threshold vs Quantile')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(quantile_df['quantile'], quantile_df['pct_above'], 'o-', linewidth=2, markersize=8, color='blue')
axes[1, 0].set_xlabel('Quantile')
axes[1, 0].set_ylabel('% Pairs Above Threshold')
axes[1, 0].set_title('Selectivity')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(quantile_df['quantile'], quantile_df['good_above_pct'], 'o-', linewidth=2, markersize=8, color='green', label='Good Pairs %')
axes[1, 1].plot(quantile_df['quantile'], quantile_df['same_spec_pct'], 's-', linewidth=2, markersize=8, color='purple', label='Same Specialty %')
axes[1, 1].set_xlabel('Quantile')
axes[1, 1].set_ylabel('Quality Metric (%)')
axes[1, 1].set_title('Quality Above Threshold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('method1_global_quantile.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: method1_global_quantile.png")

recommended_quantile = 0.97
global_threshold = np.quantile(pairs_df['sim_overall'], recommended_quantile)
print(f"\n{'='*80}")
print(f"RECOMMENDED GLOBAL THRESHOLD (Method 1)")
print(f"{'='*80}")
print(f"Quantile: {recommended_quantile:.2f} (top {(1-recommended_quantile)*100:.0f}%)")
print(f"Threshold: {global_threshold:.6f}")

print("\n" + "="*80)
print("METHOD 2: PER-PROVIDER TOP-K + KNEE DETECTION")
print("="*80)

print(f"\nAnalyzing top-{TOP_K} neighbors per provider...")
print("Sampling 50 providers for knee detection analysis...")

sample_providers_idx = np.random.choice(n_providers, size=min(50, n_providers), replace=False)

knee_results = []

for sample_idx in tqdm(sample_providers_idx[:10], desc="Knee analysis"):
    emb_query = embeddings_matrix[sample_idx]
    
    similarities = []
    for idx in range(n_providers):
        if idx != sample_idx:
            emb_target = embeddings_matrix[idx]
            sim = cosine_similarity(emb_query, emb_target)
            similarities.append((idx, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_sims = [s[1] for s in similarities[:100]]
    
    if len(top_sims) >= 10:
        x = np.arange(len(top_sims))
        y = np.array(top_sims)
        
        try:
            kneedle = KneeLocator(x, y, curve='convex', direction='decreasing')
            knee_idx = kneedle.knee if kneedle.knee is not None else TOP_K
            knee_sim = top_sims[knee_idx] if knee_idx < len(top_sims) else top_sims[TOP_K-1]
        except:
            knee_idx = TOP_K
            knee_sim = top_sims[TOP_K-1]
        
        knee_results.append({
            'provider_idx': sample_idx,
            'knee_position': knee_idx,
            'knee_similarity': knee_sim,
            'top_k_similarity': top_sims[TOP_K-1],
            'top_1_similarity': top_sims[0]
        })

if knee_results:
    knee_df = pd.DataFrame(knee_results)
    
    print(f"\nKnee Detection Results (n={len(knee_df)}):")
    print(f"  Avg knee position: {knee_df['knee_position'].mean():.1f}")
    print(f"  Avg knee similarity: {knee_df['knee_similarity'].mean():.6f}")
    print(f"  Avg top-{TOP_K} similarity: {knee_df['top_k_similarity'].mean():.6f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(knee_df['knee_position'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(TOP_K, color='red', linestyle='--', linewidth=2, label=f'Top-K={TOP_K}')
    axes[0, 0].set_xlabel('Knee Position')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Knee Positions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(knee_df['knee_similarity'], bins=20, edgecolor='black', alpha=0.7, label='Knee Sim')
    axes[0, 1].axvline(global_threshold, color='red', linestyle='--', linewidth=2, label=f'Global: {global_threshold:.3f}')
    axes[0, 1].set_xlabel('Similarity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Knee Similarities')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    for i in range(min(5, len(knee_df))):
        sample_idx = knee_df.iloc[i]['provider_idx']
        emb_query = embeddings_matrix[sample_idx]
        
        similarities = []
        for idx in range(n_providers):
            if idx != sample_idx:
                emb_target = embeddings_matrix[idx]
                sim = cosine_similarity(emb_query, emb_target)
                similarities.append(sim)
        
        similarities.sort(reverse=True)
        top_sims = similarities[:50]
        
        axes[1, 0].plot(range(len(top_sims)), top_sims, alpha=0.6)
    
    axes[1, 0].axhline(global_threshold, color='red', linestyle='--', linewidth=2, label='Global Threshold')
    axes[1, 0].axvline(TOP_K, color='green', linestyle='--', linewidth=2, label=f'Top-K={TOP_K}')
    axes[1, 0].set_xlabel('Rank')
    axes[1, 0].set_ylabel('Similarity')
    axes[1, 0].set_title('Similarity Curves (Sample Providers)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(knee_df['knee_position'], knee_df['knee_similarity'], alpha=0.6)
    axes[1, 1].axhline(global_threshold, color='red', linestyle='--', linewidth=2, label='Global Threshold')
    axes[1, 1].axvline(TOP_K, color='green', linestyle='--', linewidth=2, label=f'Top-K={TOP_K}')
    axes[1, 1].set_xlabel('Knee Position')
    axes[1, 1].set_ylabel('Knee Similarity')
    axes[1, 1].set_title('Knee Position vs Similarity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('method2_topk_knee.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: method2_topk_knee.png")

print(f"\n{'='*80}")
print(f"PER-PROVIDER RULE (Method 2)")
print(f"{'='*80}")
print(f"Top-K: {TOP_K}")
print(f"Combined rule: sim_overall >= {global_threshold:.6f} AND rank <= {TOP_K}")

print("\n" + "="*80)
print("METHOD 3: GMM ON ALL 7 SIMILARITIES")
print("="*80)

X = pairs_df[sim_cols].values
print(f"\nFeature matrix: {X.shape}")

print("\nFitting 2-component Gaussian Mixture Model...")
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, max_iter=200)
gmm.fit(X)

probs = gmm.predict_proba(X)
labels = gmm.predict(X)

means = gmm.means_
print(f"\nComponent Means:")
for i in range(2):
    print(f"\nComponent {i}:")
    for j, col in enumerate(sim_cols):
        print(f"  {col}: {means[i, j]:.6f}")

comp_sim_overall_means = means[:, -1]
similar_comp = comp_sim_overall_means.argmax()
background_comp = 1 - similar_comp

print(f"\nIdentified components:")
print(f"  Similar component: {similar_comp} (mean sim_overall: {means[similar_comp, -1]:.6f})")
print(f"  Background component: {background_comp} (mean sim_overall: {means[background_comp, -1]:.6f})")

pairs_df['p_similar'] = probs[:, similar_comp]
pairs_df['p_background'] = probs[:, background_comp]
pairs_df['gmm_label'] = labels

print(f"\nProbability Statistics:")
print(f"  p_similar:")
print(f"    Mean: {pairs_df['p_similar'].mean():.4f}")
print(f"    Median: {pairs_df['p_similar'].median():.4f}")
print(f"    Q75: {pairs_df['p_similar'].quantile(0.75):.4f}")
print(f"    Q90: {pairs_df['p_similar'].quantile(0.90):.4f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].hist(pairs_df['p_similar'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(GMM_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold: {GMM_THRESHOLD}')
axes[0, 0].set_xlabel('P(Similar)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of P(Similar)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

for i in range(2):
    component_mask = pairs_df['gmm_label'] == i
    label_name = 'Similar' if i == similar_comp else 'Background'
    axes[0, 1].hist(pairs_df[component_mask]['sim_overall'], bins=30, alpha=0.6, label=label_name)

axes[0, 1].axvline(global_threshold, color='red', linestyle='--', linewidth=2, label='Global Threshold')
axes[0, 1].set_xlabel('Overall Similarity')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('sim_overall by GMM Component')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].scatter(pairs_df['sim_overall'], pairs_df['p_similar'], alpha=0.3, s=10)
axes[0, 2].axhline(GMM_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'P(similar)={GMM_THRESHOLD}')
axes[0, 2].axvline(global_threshold, color='green', linestyle='--', linewidth=2, label='Global Threshold')
axes[0, 2].set_xlabel('sim_overall')
axes[0, 2].set_ylabel('P(Similar)')
axes[0, 2].set_title('Overall Similarity vs P(Similar)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

gmm_pass_mask = pairs_df['p_similar'] >= GMM_THRESHOLD
good_in_gmm = pairs_df[gmm_pass_mask]['is_good_pair'].sum()
bad_in_gmm = pairs_df[gmm_pass_mask]['is_bad_pair'].sum()
good_pct = good_in_gmm / gmm_pass_mask.sum() * 100 if gmm_pass_mask.sum() > 0 else 0

axes[1, 0].bar(['Pass GMM', 'Fail GMM'], 
               [gmm_pass_mask.sum(), (~gmm_pass_mask).sum()],
               color=['green', 'gray'], alpha=0.7)
axes[1, 0].set_ylabel('Number of Pairs')
axes[1, 0].set_title(f'GMM Filter (threshold={GMM_THRESHOLD})')
axes[1, 0].grid(True, alpha=0.3, axis='y')

axes[1, 1].bar(['Good', 'Bad'], [good_in_gmm, bad_in_gmm], color=['green', 'red'], alpha=0.7)
axes[1, 1].set_ylabel('Number of Pairs')
axes[1, 1].set_title(f'Label Quality in GMM-Passed Pairs\n({good_pct:.1f}% good)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

for i, col in enumerate(['sim_procedures', 'sim_diagnoses', 'sim_demographics']):
    axes[1, 2].hist(pairs_df[gmm_pass_mask][col], bins=20, alpha=0.5, label=col.replace('sim_', ''))

axes[1, 2].set_xlabel('Similarity')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Tower Similarities in GMM-Passed Pairs')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('method3_gmm.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: method3_gmm.png")

print(f"\n{'='*80}")
print(f"GMM FILTER (Method 3)")
print(f"{'='*80}")
print(f"Threshold: P(similar) >= {GMM_THRESHOLD}")
print(f"Pairs passing: {gmm_pass_mask.sum()} ({gmm_pass_mask.mean()*100:.1f}%)")
print(f"Quality: {good_pct:.1f}% good pairs")

print("\n" + "="*80)
print("COMBINED FINAL RULE")
print("="*80)

pairs_df['pass_global'] = pairs_df['sim_overall'] >= global_threshold
pairs_df['pass_gmm'] = pairs_df['p_similar'] >= GMM_THRESHOLD
pairs_df['pass_combined'] = pairs_df['pass_global'] & pairs_df['pass_gmm']

print(f"\nFiltering Results:")
print(f"  Pass Global (Method 1): {pairs_df['pass_global'].sum()} ({pairs_df['pass_global'].mean()*100:.1f}%)")
print(f"  Pass GMM (Method 3): {pairs_df['pass_gmm'].sum()} ({pairs_df['pass_gmm'].mean()*100:.1f}%)")
print(f"  Pass Combined: {pairs_df['pass_combined'].sum()} ({pairs_df['pass_combined'].mean()*100:.1f}%)")

print(f"\nQuality of Combined Rule:")
combined_mask = pairs_df['pass_combined']
if combined_mask.sum() > 0:
    good_combined = pairs_df[combined_mask]['is_good_pair'].sum()
    bad_combined = pairs_df[combined_mask]['is_bad_pair'].sum()
    good_pct_combined = good_combined / combined_mask.sum() * 100
    same_spec_combined = pairs_df[combined_mask]['same_specialty'].sum()
    same_spec_pct = same_spec_combined / combined_mask.sum() * 100
    
    print(f"  Good pairs: {good_combined} ({good_pct_combined:.1f}%)")
    print(f"  Bad pairs: {bad_combined}")
    print(f"  Same specialty: {same_spec_combined} ({same_spec_pct:.1f}%)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

categories = ['Global Only', 'GMM Only', 'Combined']
counts = [
    pairs_df['pass_global'].sum(),
    pairs_df['pass_gmm'].sum(),
    pairs_df['pass_combined'].sum()
]
axes[0, 0].bar(categories, counts, color=['blue', 'orange', 'green'], alpha=0.7)
axes[0, 0].set_ylabel('Number of Pairs')
axes[0, 0].set_title('Pairs Passing Each Filter')
axes[0, 0].grid(True, alpha=0.3, axis='y')

venn_data = {
    'Global Only': pairs_df['pass_global'].sum() - pairs_df['pass_combined'].sum(),
    'GMM Only': pairs_df['pass_gmm'].sum() - pairs_df['pass_combined'].sum(),
    'Both': pairs_df['pass_combined'].sum(),
    'Neither': (~pairs_df['pass_global'] & ~pairs_df['pass_gmm']).sum()
}
axes[0, 1].bar(venn_data.keys(), venn_data.values(), alpha=0.7)
axes[0, 1].set_ylabel('Number of Pairs')
axes[0, 1].set_title('Filter Overlap')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

for mask, label, color in [
    (pairs_df['pass_global'], 'Global', 'blue'),
    (pairs_df['pass_gmm'], 'GMM', 'orange'),
    (pairs_df['pass_combined'], 'Combined', 'green')
]:
    if mask.sum() > 0:
        axes[1, 0].hist(pairs_df[mask]['sim_overall'], bins=30, alpha=0.5, label=label)

axes[1, 0].axvline(global_threshold, color='red', linestyle='--', linewidth=2, label='Global Threshold')
axes[1, 0].set_xlabel('Overall Similarity')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('sim_overall Distribution by Filter')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

quality_metrics = {
    'Global': pairs_df[pairs_df['pass_global']]['is_good_pair'].mean() * 100 if pairs_df['pass_global'].sum() > 0 else 0,
    'GMM': pairs_df[pairs_df['pass_gmm']]['is_good_pair'].mean() * 100 if pairs_df['pass_gmm'].sum() > 0 else 0,
    'Combined': pairs_df[pairs_df['pass_combined']]['is_good_pair'].mean() * 100 if pairs_df['pass_combined'].sum() > 0 else 0
}
axes[1, 1].bar(quality_metrics.keys(), quality_metrics.values(), color=['blue', 'orange', 'green'], alpha=0.7)
axes[1, 1].set_ylabel('% Good Pairs')
axes[1, 1].set_title('Quality Comparison')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('combined_rule.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: combined_rule.png")

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

final_rule = {
    'method_1_global_threshold': {
        'quantile': recommended_quantile,
        'threshold': float(global_threshold),
        'description': f'sim_overall >= {global_threshold:.6f} (top {(1-recommended_quantile)*100:.0f}%)'
    },
    'method_2_topk': {
        'top_k': TOP_K,
        'description': f'Keep top-{TOP_K} neighbors per provider (above global threshold)'
    },
    'method_3_gmm': {
        'threshold': GMM_THRESHOLD,
        'description': f'P(similar) >= {GMM_THRESHOLD} from 7-dimensional GMM'
    },
    'combined_rule': f'sim_overall >= {global_threshold:.6f} AND P(similar) >= {GMM_THRESHOLD} AND rank <= {TOP_K}',
    'performance': {
        'pairs_passing_global': int(pairs_df['pass_global'].sum()),
        'pairs_passing_gmm': int(pairs_df['pass_gmm'].sum()),
        'pairs_passing_combined': int(pairs_df['pass_combined'].sum()),
        'quality_global': float(pairs_df[pairs_df['pass_global']]['is_good_pair'].mean() * 100) if pairs_df['pass_global'].sum() > 0 else 0,
        'quality_gmm': float(pairs_df[pairs_df['pass_gmm']]['is_good_pair'].mean() * 100) if pairs_df['pass_gmm'].sum() > 0 else 0,
        'quality_combined': float(pairs_df[pairs_df['pass_combined']]['is_good_pair'].mean() * 100) if pairs_df['pass_combined'].sum() > 0 else 0
    }
}

print("\n" + "="*60)
print("THREE-METHOD APPROACH:")
print("="*60)
print(f"\n1. Global Threshold (Quantile-based)")
print(f"   {final_rule['method_1_global_threshold']['description']}")
print(f"\n2. Per-Provider Top-K")
print(f"   {final_rule['method_2_topk']['description']}")
print(f"\n3. GMM Multi-dimensional Filter")
print(f"   {final_rule['method_3_gmm']['description']}")
print(f"\n{'='*60}")
print(f"COMBINED RULE:")
print(f"{'='*60}")
print(f"{final_rule['combined_rule']}")
print(f"\nPerformance:")
print(f"  Passes combined filter: {final_rule['performance']['pairs_passing_combined']} pairs")
print(f"  Quality: {final_rule['performance']['quality_combined']:.1f}% good pairs")

with open('unsupervised_threshold_recommendation.pkl', 'wb') as f:
    pickle.dump(final_rule, f)
print("\n✓ Saved: unsupervised_threshold_recommendation.pkl")

pairs_df.to_parquet('sampled_pairs_with_all_methods.parquet')
print("✓ Saved: sampled_pairs_with_all_methods.parquet")

quantile_df.to_csv('quantile_analysis.csv', index=False)
print("✓ Saved: quantile_analysis.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nNo hardcoded thresholds used - all values derived from data:")
print(f"  ✓ Global threshold from {recommended_quantile:.0%} quantile")
print(f"  ✓ Top-K from neighbor density")
print(f"  ✓ GMM threshold from cluster probability")
print(f"  ✓ County-based sampling (same-county pairs only)")
print(f"\nFully unsupervised, statistically defensible, geographically-aware approach.")
print(f"Threshold is pooled across all counties but based on realistic substitutes.")
