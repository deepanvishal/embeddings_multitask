"""
COUNTY-LEVEL ALTERNATIVE ANALYSIS
==================================

Analyzes providers with/without alternatives at the county level.
Shows coverage gaps by geography.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SIMILARITY_THRESHOLD = 0.75
PRIMARY_SPECIALTY_FILTER = 'WHO'

print("\n" + "="*80)
print("COUNTY-LEVEL ALTERNATIVE ANALYSIS")
print(f"Threshold: {SIMILARITY_THRESHOLD}")
print(f"Primary Specialty: {PRIMARY_SPECIALTY_FILTER}")
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

all_pins = embeddings_df['PIN'].values
n_providers = len(all_pins)
print(f"Total providers: {n_providers:,}")

emb_cols = [col for col in embeddings_df.columns if col != 'PIN']
embeddings_df = embeddings_df.set_index('PIN')
embeddings_matrix = embeddings_df[emb_cols].values

print("\n" + "="*80)
print("COMPUTING ALL SAME-COUNTY PAIRS (VECTORIZED)")
print("="*80)

pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

county_to_indices = {}
for county_state, pins in county_state_to_pins.items():
    if county_state == '|':
        continue
    indices = [pin_to_idx[pin] for pin in pins if pin in pin_to_idx]
    if len(indices) >= 2:
        county_to_indices[county_state] = indices

eligible_counties = list(county_to_indices.keys())
print(f"Eligible counties: {len(eligible_counties)}")

all_pairs_data = []

for county_idx, (county_state, indices) in enumerate(tqdm(county_to_indices.items(), desc="Processing counties")):
    n = len(indices)
    
    if n < 2:
        continue
    
    county_embeddings = embeddings_matrix[indices]
    
    norms = np.linalg.norm(county_embeddings, axis=1, keepdims=True)
    normalized = county_embeddings / (norms + 1e-8)
    
    sim_matrix = np.dot(normalized, normalized.T)
    
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
            'similarity': similarity,
            'county_state': county_state,
            'specialty_a': pin_to_specialty.get(pin_a, 'Unknown'),
            'specialty_b': pin_to_specialty.get(pin_b, 'Unknown')
        })

print(f"\nComputed {len(all_pairs_data):,} same-county pairs")

pairs_df = pd.DataFrame(all_pairs_data)

print("\n" + "="*80)
print("FILTERING PRIMARY PROVIDERS")
print("="*80)

pairs_df = pairs_df[pairs_df['specialty_a'] == PRIMARY_SPECIALTY_FILTER].copy()
print(f"Pairs with {PRIMARY_SPECIALTY_FILTER} as primary: {len(pairs_df):,}")

print("\n" + "="*80)
print("APPLYING THRESHOLD")
print("="*80)

pairs_df['is_good_pair'] = pairs_df['similarity'] >= SIMILARITY_THRESHOLD

good_pairs_df = pairs_df[pairs_df['is_good_pair']].copy()

print(f"Good pairs: {len(good_pairs_df):,}")

print("\n" + "="*80)
print("PER-PROVIDER ANALYSIS")
print("="*80)

provider_good_count = good_pairs_df.groupby('pin_a').size()

all_who_providers = set(pairs_df['pin_a'].unique())

provider_alternatives = {}
for pin in all_who_providers:
    count = provider_good_count.get(pin, 0)
    provider_alternatives[pin] = count

provider_alt_df = pd.DataFrame([
    {'PIN': pin, 'n_good_alternatives': count}
    for pin, count in provider_alternatives.items()
])

provider_alt_df['county_state'] = provider_alt_df['PIN'].map(pin_to_county_state).fillna('|')
provider_alt_df['has_alternatives'] = provider_alt_df['n_good_alternatives'] > 0

print(f"\nAnalyzed {len(provider_alt_df)} {PRIMARY_SPECIALTY_FILTER} providers")

print("\n" + "="*80)
print("COUNTY-LEVEL AGGREGATION")
print("="*80)

county_summary = provider_alt_df.groupby('county_state').agg({
    'has_alternatives': ['sum', lambda x: (~x).sum(), 'count']
}).reset_index()

county_summary.columns = ['county_state', 'providers_with_alternatives', 'providers_without_alternatives', 'total_providers']

county_summary['pct_with_alternatives'] = (
    county_summary['providers_with_alternatives'] / county_summary['total_providers'] * 100
)

county_summary['pct_without_alternatives'] = (
    county_summary['providers_without_alternatives'] / county_summary['total_providers'] * 100
)

county_summary[['county_nm', 'state']] = county_summary['county_state'].str.split('|', expand=True)

county_summary = county_summary.sort_values('providers_without_alternatives', ascending=False)

print(f"\nCounties analyzed: {len(county_summary)}")
print(f"\nOverall statistics:")
print(f"  Total {PRIMARY_SPECIALTY_FILTER} providers: {county_summary['total_providers'].sum():,}")
print(f"  Providers with alternatives: {county_summary['providers_with_alternatives'].sum():,}")
print(f"  Providers without alternatives: {county_summary['providers_without_alternatives'].sum():,}")
print(f"  Coverage rate: {county_summary['providers_with_alternatives'].sum() / county_summary['total_providers'].sum() * 100:.1f}%")

print("\n" + "="*80)
print("TOP 20 COUNTIES BY PROVIDERS WITHOUT ALTERNATIVES")
print("="*80)

print(f"\n{'County':<30s} {'State':<6s} {'With Alt':<10s} {'Without Alt':<12s} {'Total':<8s} {'% Without':<10s}")
print("-" * 90)

for _, row in county_summary.head(20).iterrows():
    print(f"{row['county_nm']:<30s} {row['state']:<6s} "
          f"{int(row['providers_with_alternatives']):<10d} "
          f"{int(row['providers_without_alternatives']):<12d} "
          f"{int(row['total_providers']):<8d} "
          f"{row['pct_without_alternatives']:>6.1f}%")

print("\n" + "="*80)
print("COUNTIES WITH 100% PROVIDERS WITHOUT ALTERNATIVES")
print("="*80)

zero_coverage = county_summary[county_summary['providers_with_alternatives'] == 0].copy()
print(f"\nFound {len(zero_coverage)} counties with NO providers having alternatives")

if len(zero_coverage) > 0:
    print(f"\n{'County':<30s} {'State':<6s} {'Total Providers':<15s}")
    print("-" * 60)
    for _, row in zero_coverage.head(30).iterrows():
        print(f"{row['county_nm']:<30s} {row['state']:<6s} {int(row['total_providers']):<15d}")

print("\n" + "="*80)
print("STATE-LEVEL SUMMARY")
print("="*80)

state_summary = county_summary.groupby('state').agg({
    'providers_with_alternatives': 'sum',
    'providers_without_alternatives': 'sum',
    'total_providers': 'sum'
}).reset_index()

state_summary['pct_without_alternatives'] = (
    state_summary['providers_without_alternatives'] / state_summary['total_providers'] * 100
)

state_summary = state_summary.sort_values('providers_without_alternatives', ascending=False)

print(f"\n{'State':<6s} {'With Alt':<10s} {'Without Alt':<12s} {'Total':<8s} {'% Without':<10s}")
print("-" * 60)

for _, row in state_summary.head(20).iterrows():
    print(f"{row['state']:<6s} "
          f"{int(row['providers_with_alternatives']):<10d} "
          f"{int(row['providers_without_alternatives']):<12d} "
          f"{int(row['total_providers']):<8d} "
          f"{row['pct_without_alternatives']:>6.1f}%")

print("\n" + "="*80)
print("VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

top_20_counties = county_summary.head(20)

axes[0, 0].barh(range(len(top_20_counties)), top_20_counties['providers_without_alternatives'], 
                color='red', alpha=0.7, label='Without Alternatives')
axes[0, 0].barh(range(len(top_20_counties)), top_20_counties['providers_with_alternatives'], 
                left=top_20_counties['providers_without_alternatives'],
                color='green', alpha=0.7, label='With Alternatives')
axes[0, 0].set_yticks(range(len(top_20_counties)))
axes[0, 0].set_yticklabels([f"{row['county_nm']}, {row['state']}" for _, row in top_20_counties.iterrows()], 
                            fontsize=8)
axes[0, 0].set_xlabel('Number of Providers')
axes[0, 0].set_title(f'Top 20 Counties by Providers Without Alternatives\n({PRIMARY_SPECIALTY_FILTER} specialty)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='x')
axes[0, 0].invert_yaxis()

axes[0, 1].scatter(county_summary['total_providers'], 
                   county_summary['pct_without_alternatives'],
                   alpha=0.6, s=50)
axes[0, 1].set_xlabel('Total Providers in County')
axes[0, 1].set_ylabel('% Without Alternatives')
axes[0, 1].set_title('Coverage vs County Size')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
axes[0, 1].legend()

coverage_bins = pd.cut(county_summary['pct_without_alternatives'], 
                       bins=[0, 10, 25, 50, 75, 100],
                       labels=['0-10%', '10-25%', '25-50%', '50-75%', '75-100%'])
coverage_dist = coverage_bins.value_counts().sort_index()

axes[1, 0].bar(range(len(coverage_dist)), coverage_dist.values, 
               color=['green', 'yellowgreen', 'yellow', 'orange', 'red'], alpha=0.7)
axes[1, 0].set_xticks(range(len(coverage_dist)))
axes[1, 0].set_xticklabels(coverage_dist.index, rotation=45)
axes[1, 0].set_xlabel('% Providers Without Alternatives')
axes[1, 0].set_ylabel('Number of Counties')
axes[1, 0].set_title('Distribution of Coverage Across Counties')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(coverage_dist.values):
    axes[1, 0].text(i, val, str(val), ha='center', va='bottom')

top_10_states = state_summary.head(10)
x = range(len(top_10_states))
axes[1, 1].bar(x, top_10_states['providers_without_alternatives'], 
               color='red', alpha=0.7, label='Without Alternatives')
axes[1, 1].bar(x, top_10_states['providers_with_alternatives'], 
               bottom=top_10_states['providers_without_alternatives'],
               color='green', alpha=0.7, label='With Alternatives')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(top_10_states['state'], rotation=45)
axes[1, 1].set_ylabel('Number of Providers')
axes[1, 1].set_title(f'Top 10 States by Providers Without Alternatives\n({PRIMARY_SPECIALTY_FILTER} specialty)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'county_analysis_{PRIMARY_SPECIALTY_FILTER}_threshold_{SIMILARITY_THRESHOLD}.png', 
            dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: county_analysis_{PRIMARY_SPECIALTY_FILTER}_threshold_{SIMILARITY_THRESHOLD}.png")

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

county_summary.to_csv(f'county_level_alternatives_{PRIMARY_SPECIALTY_FILTER}_threshold_{SIMILARITY_THRESHOLD}.csv', 
                      index=False)
print(f"✓ Saved: county_level_alternatives_{PRIMARY_SPECIALTY_FILTER}_threshold_{SIMILARITY_THRESHOLD}.csv")

state_summary.to_csv(f'state_level_alternatives_{PRIMARY_SPECIALTY_FILTER}_threshold_{SIMILARITY_THRESHOLD}.csv', 
                     index=False)
print(f"✓ Saved: state_level_alternatives_{PRIMARY_SPECIALTY_FILTER}_threshold_{SIMILARITY_THRESHOLD}.csv")

provider_alt_df.to_csv(f'provider_level_alternatives_{PRIMARY_SPECIALTY_FILTER}_threshold_{SIMILARITY_THRESHOLD}.csv', 
                       index=False)
print(f"✓ Saved: provider_level_alternatives_{PRIMARY_SPECIALTY_FILTER}_threshold_{SIMILARITY_THRESHOLD}.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nKey Findings:")
print(f"  - {len(county_summary)} counties analyzed")
print(f"  - {len(zero_coverage)} counties with 0% coverage")
print(f"  - {county_summary['providers_without_alternatives'].sum():,} providers without alternatives")
print(f"  - Overall coverage: {county_summary['providers_with_alternatives'].sum() / county_summary['total_providers'].sum() * 100:.1f}%")
