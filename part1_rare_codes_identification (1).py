import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("PART 1: RARE CODE IDENTIFICATION WITH COMPOSITE RARITY SCORE")
print("="*80)

CUTOFF_PERCENTILE = 10

print(f"\nConfiguration: CUTOFF_PERCENTILE = {CUTOFF_PERCENTILE}")

print("\n" + "="*80)
print("STEP 1: LOAD AND UNDERSTAND DATA")
print("="*80)

procedure_df = pd.read_parquet('procedure_df.parquet')

print(f"\nDataset Overview:")
print(f"  Total rows: {len(procedure_df):,}")
print(f"  Unique PINs (Providers): {procedure_df['PIN'].nunique():,}")
print(f"  Unique codes: {procedure_df['code'].nunique():,}")
print(f"  Total claims: {procedure_df['claims'].sum():,.0f}")

print(f"\nSample data:")
print(procedure_df.head(10))

print("\n" + "="*80)
print("STEP 2: CODE-LEVEL STATISTICS")
print("="*80)

code_stats = procedure_df.groupby('code').agg({
    'claims': 'sum',
    'PIN': 'nunique'
}).rename(columns={
    'claims': 'total_claims',
    'PIN': 'num_providers'
}).reset_index()

total_providers = procedure_df['PIN'].nunique()

print(f"\nTotal claims per code:")
print(code_stats['total_claims'].describe())

print(f"\nProviders per code:")
print(code_stats['num_providers'].describe())

print(f"\nPercentile breakdown - Volume:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = code_stats['total_claims'].quantile(p/100)
    count = len(code_stats[code_stats['total_claims'] <= val])
    print(f"  P{p:2d}: <= {val:7.0f} claims ({count:5d} codes)")

print(f"\nPercentile breakdown - Providers:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = code_stats['num_providers'].quantile(p/100)
    count = len(code_stats[code_stats['num_providers'] <= val])
    print(f"  P{p:2d}: <= {val:5.0f} providers ({count:5d} codes)")

print("\n" + "="*80)
print("STEP 3: CALCULATE RARITY SCORE COMPONENTS")
print("="*80)

print("\nComponent 1: Herfindahl Index (Provider Concentration)")
print("  - Measures how concentrated a service is among providers")
print("  - Range: 1/N (distributed) to 1 (monopoly)")
print("  - Higher = more concentrated")

herfindahl_scores = []
for code in code_stats['code']:
    code_data = procedure_df[procedure_df['code'] == code]
    total_claims = code_data['claims'].sum()
    shares = (code_data['claims'] / total_claims) ** 2
    herfindahl = shares.sum()
    herfindahl_scores.append(herfindahl)

code_stats['herfindahl'] = herfindahl_scores

print(f"\nHerfindahl Index Statistics:")
print(code_stats['herfindahl'].describe())

print("\nComponent 2: Provider Scarcity")
print("  - How few providers offer this service")
print("  - Formula: 1 - (num_providers / total_providers)")
print("  - Range: 0 (everyone) to 1 (one provider)")

code_stats['provider_scarcity'] = 1 - (code_stats['num_providers'] / total_providers)

print(f"\nProvider Scarcity Statistics:")
print(code_stats['provider_scarcity'].describe())

print("\nComponent 3: Inverse Prevalence")
print("  - How uncommon is this service in volume distribution")
print("  - Formula: 1 - (percentile_rank / 100)")
print("  - Range: 0 (very common) to 1 (very rare)")

code_stats['volume_percentile'] = code_stats['total_claims'].rank(pct=True) * 100
code_stats['inverse_prevalence'] = 1 - (code_stats['volume_percentile'] / 100)

print(f"\nInverse Prevalence Statistics:")
print(code_stats['inverse_prevalence'].describe())

print("\n" + "="*80)
print("STEP 4: COMPOSITE RARITY SCORE")
print("="*80)

print("\nCalculating Composite Rarity Score:")
print("  Formula: (Herfindahl × Scarcity × Inverse_Prevalence) ^ (1/3)")
print("  Geometric mean ensures all three components matter")

code_stats['rarity_score'] = (
    code_stats['herfindahl'] * 
    code_stats['provider_scarcity'] * 
    code_stats['inverse_prevalence']
) ** (1/3)

print(f"\nRarity Score Statistics:")
print(code_stats['rarity_score'].describe())

print(f"\nRarity Score Percentiles:")
for p in [50, 75, 90, 95, 99]:
    val = code_stats['rarity_score'].quantile(p/100)
    count = len(code_stats[code_stats['rarity_score'] >= val])
    print(f"  P{p:2d}: >= {val:.6f} ({count:5d} codes)")

print(f"\nTop 20 codes by Rarity Score:")
top_rare = code_stats.sort_values('rarity_score', ascending=False).head(20)
print(top_rare[['code', 'total_claims', 'num_providers', 'herfindahl', 'provider_scarcity', 'inverse_prevalence', 'rarity_score']])

print(f"\nBottom 20 codes by Rarity Score (Common services):")
bottom_rare = code_stats.sort_values('rarity_score', ascending=True).head(20)
print(bottom_rare[['code', 'total_claims', 'num_providers', 'herfindahl', 'provider_scarcity', 'inverse_prevalence', 'rarity_score']])

print("\n" + "="*80)
print("STEP 5: LOG-TRANSFORMED DISTRIBUTIONS")
print("="*80)

code_stats['log_claims'] = np.log1p(code_stats['total_claims'])
code_stats['log_providers'] = np.log1p(code_stats['num_providers'])

print(f"\nLog-transformed Volume Statistics:")
print(code_stats['log_claims'].describe())

print(f"\nLog-transformed Providers Statistics:")
print(code_stats['log_providers'].describe())

print(f"\nLog-transformed Volume Percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    log_val = code_stats['log_claims'].quantile(p/100)
    actual_val = np.expm1(log_val)
    print(f"  P{p:2d}: log={log_val:.4f}, actual={actual_val:.0f} claims")

print(f"\nLog-transformed Providers Percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    log_val = code_stats['log_providers'].quantile(p/100)
    actual_val = np.expm1(log_val)
    print(f"  P{p:2d}: log={log_val:.4f}, actual={actual_val:.0f} providers")

print("\n" + "="*80)
print("STEP 6: IDENTIFY RARE CODES")
print("="*80)

rarity_threshold = code_stats['rarity_score'].quantile(1 - CUTOFF_PERCENTILE/100)

print(f"\nUsing P{100-CUTOFF_PERCENTILE} (top {CUTOFF_PERCENTILE}% rarest codes)")
print(f"  Rarity score threshold: >= {rarity_threshold:.6f}")

rare_codes_df = code_stats[code_stats['rarity_score'] >= rarity_threshold].copy()
rare_codes_df = rare_codes_df.sort_values('rarity_score', ascending=False)

print(f"\nRare codes identified: {len(rare_codes_df):,}")
print(f"Percentage of total codes: {100*len(rare_codes_df)/len(code_stats):.2f}%")

print(f"\nRare codes characteristics:")
print(f"  Volume range: {rare_codes_df['total_claims'].min():.0f} to {rare_codes_df['total_claims'].max():.0f}")
print(f"  Provider range: {rare_codes_df['num_providers'].min():.0f} to {rare_codes_df['num_providers'].max():.0f}")
print(f"  Median volume: {rare_codes_df['total_claims'].median():.0f}")
print(f"  Median providers: {rare_codes_df['num_providers'].median():.0f}")

print(f"\nTop 30 rare codes:")
print(rare_codes_df.head(30)[['code', 'total_claims', 'num_providers', 'rarity_score']])

print("\n" + "="*80)
print("STEP 7: VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(3, 3, figsize=(20, 18))

axes[0, 0].hist(code_stats['log_claims'], bins=50, edgecolor='black', alpha=0.7)
log_volume_threshold = np.log1p(code_stats[code_stats['rarity_score'] >= rarity_threshold]['total_claims'].max())
axes[0, 0].axvline(log_volume_threshold, color='red', linestyle='--', linewidth=2, label=f'Rare codes max')
axes[0, 0].set_xlabel('Log(Total Claims + 1)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Code Volume (Log Scale)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(code_stats['log_providers'], bins=50, edgecolor='black', alpha=0.7)
log_provider_threshold = np.log1p(code_stats[code_stats['rarity_score'] >= rarity_threshold]['num_providers'].max())
axes[0, 1].axvline(log_provider_threshold, color='red', linestyle='--', linewidth=2, label=f'Rare codes max')
axes[0, 1].set_xlabel('Log(Number of Providers + 1)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Providers per Code (Log Scale)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].scatter(
    code_stats['num_providers'], 
    code_stats['total_claims'],
    alpha=0.3, s=20, label='All codes', c='blue'
)
axes[0, 2].scatter(
    rare_codes_df['num_providers'], 
    rare_codes_df['total_claims'],
    alpha=0.7, s=30, label='Rare codes', c='red'
)
axes[0, 2].set_xlabel('Number of Providers')
axes[0, 2].set_ylabel('Total Claims')
axes[0, 2].set_title('Code Volume vs Provider Count')
axes[0, 2].set_yscale('log')
axes[0, 2].set_xscale('log')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].hist(code_stats['herfindahl'], bins=50, edgecolor='black', alpha=0.7, label='All codes')
axes[1, 0].hist(rare_codes_df['herfindahl'], bins=50, edgecolor='black', alpha=0.7, color='red', label='Rare codes')
axes[1, 0].set_xlabel('Herfindahl Index')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Provider Concentration (Herfindahl)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(code_stats['provider_scarcity'], bins=50, edgecolor='black', alpha=0.7, label='All codes')
axes[1, 1].hist(rare_codes_df['provider_scarcity'], bins=50, edgecolor='black', alpha=0.7, color='red', label='Rare codes')
axes[1, 1].set_xlabel('Provider Scarcity')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Provider Scarcity Score')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].hist(code_stats['inverse_prevalence'], bins=50, edgecolor='black', alpha=0.7, label='All codes')
axes[1, 2].hist(rare_codes_df['inverse_prevalence'], bins=50, edgecolor='black', alpha=0.7, color='red', label='Rare codes')
axes[1, 2].set_xlabel('Inverse Prevalence')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Inverse Prevalence Score')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

axes[2, 0].hist(code_stats['rarity_score'], bins=100, edgecolor='black', alpha=0.7, label='All codes')
axes[2, 0].axvline(rarity_threshold, color='red', linestyle='--', linewidth=2, label=f'P{100-CUTOFF_PERCENTILE} threshold')
axes[2, 0].set_xlabel('Composite Rarity Score')
axes[2, 0].set_ylabel('Frequency')
axes[2, 0].set_title('Composite Rarity Score Distribution')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].scatter(
    code_stats['herfindahl'],
    code_stats['provider_scarcity'],
    c=code_stats['rarity_score'],
    cmap='RdYlBu_r',
    alpha=0.6,
    s=20
)
axes[2, 1].scatter(
    rare_codes_df['herfindahl'],
    rare_codes_df['provider_scarcity'],
    edgecolor='black',
    linewidth=1,
    facecolor='none',
    s=50,
    label='Rare codes'
)
axes[2, 1].set_xlabel('Herfindahl Index')
axes[2, 1].set_ylabel('Provider Scarcity')
axes[2, 1].set_title('Concentration vs Scarcity')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)
plt.colorbar(axes[2, 1].collections[0], ax=axes[2, 1], label='Rarity Score')

axes[2, 2].scatter(
    code_stats['total_claims'],
    code_stats['rarity_score'],
    alpha=0.3, s=20, label='All codes'
)
axes[2, 2].scatter(
    rare_codes_df['total_claims'],
    rare_codes_df['rarity_score'],
    alpha=0.7, s=30, color='red', label='Rare codes'
)
axes[2, 2].axhline(rarity_threshold, color='red', linestyle='--', linewidth=1)
axes[2, 2].set_xlabel('Total Claims')
axes[2, 2].set_ylabel('Rarity Score')
axes[2, 2].set_title('Volume vs Rarity Score')
axes[2, 2].set_xscale('log')
axes[2, 2].legend()
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('part1_rarity_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved: part1_rarity_analysis.png")
plt.close()

print("\n" + "="*80)
print("STEP 8: SAVE RESULTS")
print("="*80)

output_columns = ['code', 'total_claims', 'num_providers', 'herfindahl', 
                  'provider_scarcity', 'inverse_prevalence', 'rarity_score']
rare_codes_output = rare_codes_df[output_columns].copy()

rare_codes_output.to_parquet('rare_codes.parquet', index=False)

print(f"\nSaved: rare_codes.parquet")
print(f"  Columns: {list(rare_codes_output.columns)}")
print(f"  Shape: {rare_codes_output.shape}")

code_stats.to_parquet('all_codes_with_rarity.parquet', index=False)
print(f"\nSaved: all_codes_with_rarity.parquet (for reference)")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  Total codes: {len(code_stats):,}")
print(f"  Rare codes (top {CUTOFF_PERCENTILE}%): {len(rare_codes_df):,}")
print(f"  Rarity threshold: {rarity_threshold:.6f}")
print(f"\nRarity Score Components:")
print(f"  1. Herfindahl Index: Provider concentration")
print(f"  2. Provider Scarcity: How few providers offer service")
print(f"  3. Inverse Prevalence: How uncommon in volume distribution")
print(f"  Composite: Geometric mean of all three")
print(f"\nFiles created:")
print(f"  - rare_codes.parquet")
print(f"  - all_codes_with_rarity.parquet")
print(f"  - part1_rarity_analysis.png")
print(f"\nTo change cutoff: modify CUTOFF_PERCENTILE = {CUTOFF_PERCENTILE}")
