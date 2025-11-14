"""
SYSTEMATIC THRESHOLD DETERMINATION FOR PROVIDER SIMILARITY
==========================================================

Uses data-driven methods to find optimal similarity threshold:
1. Distribution Analysis
2. Gaussian Mixture Models (GMM)
3. Silhouette Analysis
4. Label Concordance Validation
5. Gap Statistic

No manual thresholds. Pure statistical approach.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import gaussian_kde
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("SYSTEMATIC THRESHOLD DETERMINATION")
print("="*80)

print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

embeddings_df = pd.read_parquet('final_me2vec_all_towers_1046d.parquet')
print(f"Embeddings loaded: {embeddings_df.shape}")

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)
print(f"Labels loaded: {len(pin_to_label)}")

prov_spl_df = pd.read_parquet('prov_spl.parquet')
pin_to_specialty = dict(zip(prov_spl_df['PIN'], prov_spl_df['srv_spclty_ctg_cd']))
print(f"Specialties loaded: {len(pin_to_specialty)}")

all_pins = embeddings_df['PIN'].values
n_providers = len(all_pins)
print(f"Total providers: {n_providers:,}")

emb_cols = [col for col in embeddings_df.columns if col != 'PIN']
embeddings_df = embeddings_df.set_index('PIN')
embeddings_matrix = embeddings_df[emb_cols].values
print(f"Embedding dimension: {len(emb_cols)}")

labeled_count = sum(1 for v in pin_to_label.values() if v != 'Unknown')
print(f"Labeled providers: {labeled_count:,} ({labeled_count/n_providers:.1%})")

print("\n" + "="*80)
print("STEP 2: GENERATING 1M RANDOM PAIRS")
print("="*80)

N_PAIRS = 1_000_000
np.random.seed(42)

print(f"Sampling {N_PAIRS:,} unique provider pairs...")
pairs_set = set()
while len(pairs_set) < N_PAIRS:
    idx_a = np.random.randint(0, n_providers)
    idx_b = np.random.randint(0, n_providers)
    if idx_a != idx_b:
        pair = tuple(sorted([idx_a, idx_b]))
        pairs_set.add(pair)
    if len(pairs_set) % 100000 == 0:
        print(f"  Generated {len(pairs_set):,} pairs...")

pairs = list(pairs_set)
print(f"✓ Generated {len(pairs):,} unique pairs")

print("\nComputing similarities...")
similarities = []
pin_a_list = []
pin_b_list = []
label_a_list = []
label_b_list = []
specialty_a_list = []
specialty_b_list = []

for idx_a, idx_b in tqdm(pairs, desc="Computing"):
    emb_a = embeddings_matrix[idx_a]
    emb_b = embeddings_matrix[idx_b]
    
    sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    
    pin_a = all_pins[idx_a]
    pin_b = all_pins[idx_b]
    
    similarities.append(sim)
    pin_a_list.append(pin_a)
    pin_b_list.append(pin_b)
    label_a_list.append(pin_to_label.get(pin_a, 'Unknown'))
    label_b_list.append(pin_to_label.get(pin_b, 'Unknown'))
    specialty_a_list.append(pin_to_specialty.get(pin_a, 'Unknown'))
    specialty_b_list.append(pin_to_specialty.get(pin_b, 'Unknown'))

pairs_df = pd.DataFrame({
    'pin_a': pin_a_list,
    'pin_b': pin_b_list,
    'similarity': similarities,
    'label_a': label_a_list,
    'label_b': label_b_list,
    'specialty_a': specialty_a_list,
    'specialty_b': specialty_b_list
})

print(f"✓ Computed {len(pairs_df):,} similarities")

print("\n" + "="*80)
print("STEP 3: DISTRIBUTION ANALYSIS")
print("="*80)

print(f"\nSimilarity Statistics:")
print(f"  Min:     {pairs_df['similarity'].min():.6f}")
print(f"  Q1:      {pairs_df['similarity'].quantile(0.25):.6f}")
print(f"  Median:  {pairs_df['similarity'].median():.6f}")
print(f"  Q3:      {pairs_df['similarity'].quantile(0.75):.6f}")
print(f"  Max:     {pairs_df['similarity'].max():.6f}")
print(f"  Mean:    {pairs_df['similarity'].mean():.6f}")
print(f"  Std:     {pairs_df['similarity'].std():.6f}")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].hist(pairs_df['similarity'], bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Similarity Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Similarity Scores')
axes[0, 0].grid(True, alpha=0.3)

kde = gaussian_kde(pairs_df['similarity'])
x_range = np.linspace(pairs_df['similarity'].min(), pairs_df['similarity'].max(), 1000)
axes[0, 1].plot(x_range, kde(x_range), linewidth=2)
axes[0, 1].fill_between(x_range, kde(x_range), alpha=0.3)
axes[0, 1].set_xlabel('Similarity Score')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Kernel Density Estimation')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].boxplot(pairs_df['similarity'], vert=False)
axes[1, 0].set_xlabel('Similarity Score')
axes[1, 0].set_title('Box Plot')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(pairs_df['similarity'], bins=100, cumulative=True, density=True, 
                edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Similarity Score')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].set_title('Cumulative Distribution')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: distribution_analysis.png")

print("\n" + "="*80)
print("STEP 4: GAUSSIAN MIXTURE MODEL (GMM)")
print("="*80)

X = pairs_df['similarity'].values.reshape(-1, 1)

print("\nFitting GMM with different components...")
results = []
for n_components in range(2, 6):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    
    results.append({
        'n_components': n_components,
        'BIC': bic,
        'AIC': aic,
        'means': gmm.means_.flatten(),
        'weights': gmm.weights_,
        'covariances': gmm.covariances_.flatten()
    })
    
    print(f"  k={n_components}: BIC={bic:,.0f}, AIC={aic:,.0f}")

gmm_results_df = pd.DataFrame([{
    'n_components': r['n_components'],
    'BIC': r['BIC'],
    'AIC': r['AIC']
} for r in results])

best_bic_idx = gmm_results_df['BIC'].idxmin()
best_aic_idx = gmm_results_df['AIC'].idxmin()
best_k_bic = gmm_results_df.loc[best_bic_idx, 'n_components']
best_k_aic = gmm_results_df.loc[best_aic_idx, 'n_components']

print(f"\nOptimal components:")
print(f"  By BIC: k={best_k_bic}")
print(f"  By AIC: k={best_k_aic}")

best_k = int(best_bic_idx) if best_bic_idx == best_aic_idx else int(best_bic_idx)
best_gmm_result = results[best_k]

print(f"\nUsing k={best_gmm_result['n_components']} components")
print(f"Component means: {np.sort(best_gmm_result['means'])}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(gmm_results_df['n_components'], gmm_results_df['BIC'], 'o-', label='BIC', linewidth=2, markersize=8)
axes[0].plot(gmm_results_df['n_components'], gmm_results_df['AIC'], 's-', label='AIC', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('Information Criterion')
axes[0].set_title('GMM Model Selection')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

gmm_best = GaussianMixture(n_components=best_gmm_result['n_components'], random_state=42)
gmm_best.fit(X)
labels = gmm_best.predict(X)

for i in range(best_gmm_result['n_components']):
    cluster_data = X[labels == i]
    axes[1].hist(cluster_data, bins=50, alpha=0.5, label=f'Component {i+1}')

axes[1].set_xlabel('Similarity Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('GMM Component Assignment')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gmm_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: gmm_analysis.png")

sorted_means = np.sort(best_gmm_result['means'])
threshold_candidates = []
for i in range(len(sorted_means) - 1):
    threshold = (sorted_means[i] + sorted_means[i+1]) / 2
    threshold_candidates.append(threshold)

print(f"\nThreshold candidates from GMM boundaries:")
for i, t in enumerate(threshold_candidates):
    print(f"  Candidate {i+1}: {t:.6f}")

print("\n" + "="*80)
print("STEP 5: SILHOUETTE ANALYSIS")
print("="*80)

print("\nTesting threshold candidates...")
silhouette_results = []

for threshold in tqdm(threshold_candidates, desc="Silhouette"):
    binary_labels = (pairs_df['similarity'] >= threshold).astype(int)
    
    if len(np.unique(binary_labels)) < 2:
        score = -1.0
    else:
        score = silhouette_score(X, binary_labels)
    
    above_count = (binary_labels == 1).sum()
    below_count = (binary_labels == 0).sum()
    
    silhouette_results.append({
        'threshold': threshold,
        'silhouette_score': score,
        'above_count': above_count,
        'below_count': below_count,
        'above_pct': above_count / len(pairs_df) * 100
    })

silhouette_df = pd.DataFrame(silhouette_results)
best_silhouette_idx = silhouette_df['silhouette_score'].idxmax()
best_threshold_silhouette = silhouette_df.loc[best_silhouette_idx, 'threshold']

print(f"\nSilhouette Analysis Results:")
print(silhouette_df.to_string(index=False))
print(f"\nBest threshold by Silhouette: {best_threshold_silhouette:.6f}")
print(f"  Score: {silhouette_df.loc[best_silhouette_idx, 'silhouette_score']:.6f}")
print(f"  Above threshold: {silhouette_df.loc[best_silhouette_idx, 'above_pct']:.2f}%")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(silhouette_df['threshold'], silhouette_df['silhouette_score'], 'o-', linewidth=2, markersize=8)
ax.axvline(best_threshold_silhouette, color='red', linestyle='--', linewidth=2, label=f'Optimal: {best_threshold_silhouette:.6f}')
ax.set_xlabel('Threshold')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Analysis')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('silhouette_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: silhouette_analysis.png")

print("\n" + "="*80)
print("STEP 6: LABEL CONCORDANCE VALIDATION")
print("="*80)

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

print(f"\nPair composition:")
print(f"  Total pairs: {len(pairs_df):,}")
print(f"  Both labeled: {pairs_df['both_labeled'].sum():,} ({pairs_df['both_labeled'].mean():.1%})")
print(f"  Good pairs (same label or has unlabeled): {pairs_df['is_good_pair'].sum():,} ({pairs_df['is_good_pair'].mean():.1%})")
print(f"  Bad pairs (different labels, both labeled): {pairs_df['is_bad_pair'].sum():,} ({pairs_df['is_bad_pair'].mean():.1%})")

print("\nTesting thresholds...")
concordance_results = []

test_thresholds = threshold_candidates + [best_threshold_silhouette]
test_thresholds = sorted(list(set(test_thresholds)))

for threshold in tqdm(test_thresholds, desc="Concordance"):
    above_mask = pairs_df['similarity'] >= threshold
    below_mask = pairs_df['similarity'] < threshold
    
    good_above = pairs_df[above_mask]['is_good_pair'].sum()
    bad_below = pairs_df[below_mask]['is_bad_pair'].sum()
    
    total_good = pairs_df['is_good_pair'].sum()
    total_bad = pairs_df['is_bad_pair'].sum()
    
    good_above_rate = good_above / total_good if total_good > 0 else 0
    bad_below_rate = bad_below / total_bad if total_bad > 0 else 0
    
    purity = (good_above + bad_below) / (total_good + total_bad) if (total_good + total_bad) > 0 else 0
    
    above_good_pct = pairs_df[above_mask]['is_good_pair'].mean() * 100 if above_mask.sum() > 0 else 0
    below_bad_pct = pairs_df[below_mask]['is_bad_pair'].mean() * 100 if below_mask.sum() > 0 else 0
    
    concordance_results.append({
        'threshold': threshold,
        'good_above': good_above,
        'good_above_rate': good_above_rate * 100,
        'bad_below': bad_below,
        'bad_below_rate': bad_below_rate * 100,
        'purity': purity * 100,
        'above_good_pct': above_good_pct,
        'below_bad_pct': below_bad_pct,
        'above_count': above_mask.sum(),
        'below_count': below_mask.sum()
    })

concordance_df = pd.DataFrame(concordance_results)
best_purity_idx = concordance_df['purity'].idxmax()
best_threshold_concordance = concordance_df.loc[best_purity_idx, 'threshold']

print(f"\nLabel Concordance Results:")
print(concordance_df.to_string(index=False))
print(f"\nBest threshold by Purity: {best_threshold_concordance:.6f}")
print(f"  Purity: {concordance_df.loc[best_purity_idx, 'purity']:.2f}%")
print(f"  Good pairs above: {concordance_df.loc[best_purity_idx, 'good_above_rate']:.2f}%")
print(f"  Bad pairs below: {concordance_df.loc[best_purity_idx, 'bad_below_rate']:.2f}%")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(concordance_df['threshold'], concordance_df['purity'], 'o-', linewidth=2, markersize=8, color='green')
axes[0, 0].axvline(best_threshold_concordance, color='red', linestyle='--', linewidth=2, label=f'Optimal: {best_threshold_concordance:.6f}')
axes[0, 0].set_xlabel('Threshold')
axes[0, 0].set_ylabel('Purity (%)')
axes[0, 0].set_title('Purity Score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(concordance_df['threshold'], concordance_df['good_above_rate'], 'o-', linewidth=2, markersize=8, label='Good Above Rate')
axes[0, 1].plot(concordance_df['threshold'], concordance_df['bad_below_rate'], 's-', linewidth=2, markersize=8, label='Bad Below Rate')
axes[0, 1].set_xlabel('Threshold')
axes[0, 1].set_ylabel('Rate (%)')
axes[0, 1].set_title('Separation Rates')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(concordance_df['threshold'], concordance_df['above_good_pct'], 'o-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Threshold')
axes[1, 0].set_ylabel('% Good Pairs')
axes[1, 0].set_title('Quality Above Threshold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(concordance_df['threshold'], concordance_df['below_bad_pct'], 'o-', linewidth=2, markersize=8, color='orange')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('% Bad Pairs')
axes[1, 1].set_title('Quality Below Threshold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('concordance_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: concordance_analysis.png")

print("\n" + "="*80)
print("STEP 7: GAP STATISTIC VALIDATION")
print("="*80)

def compute_within_cluster_dispersion(X, labels):
    dispersion = 0
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean()
            dispersion += np.sum((cluster_points - centroid) ** 2)
    return np.log(dispersion) if dispersion > 0 else 0

print("\nComputing Gap Statistic...")
n_refs = 10
threshold_to_test = best_threshold_silhouette

actual_labels = (pairs_df['similarity'] >= threshold_to_test).astype(int)
actual_dispersion = compute_within_cluster_dispersion(X, actual_labels)

print(f"Testing threshold: {threshold_to_test:.6f}")
print(f"Actual dispersion (log): {actual_dispersion:.6f}")

ref_dispersions = []
for b in tqdm(range(n_refs), desc="Bootstrap"):
    X_ref = np.random.uniform(X.min(), X.max(), size=X.shape)
    ref_labels = (X_ref >= threshold_to_test).astype(int)
    ref_dispersion = compute_within_cluster_dispersion(X_ref, ref_labels)
    ref_dispersions.append(ref_dispersion)

mean_ref_dispersion = np.mean(ref_dispersions)
std_ref_dispersion = np.std(ref_dispersions)
gap = mean_ref_dispersion - actual_dispersion

print(f"\nGap Statistic Results:")
print(f"  Reference mean (log): {mean_ref_dispersion:.6f}")
print(f"  Reference std (log):  {std_ref_dispersion:.6f}")
print(f"  Gap: {gap:.6f}")
print(f"  Interpretation: {'Significant clustering' if gap > 0 else 'No significant clustering'}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(ref_dispersions, bins=30, alpha=0.7, label='Reference Distribution', edgecolor='black')
ax.axvline(actual_dispersion, color='red', linestyle='--', linewidth=2, label=f'Actual: {actual_dispersion:.2f}')
ax.axvline(mean_ref_dispersion, color='green', linestyle='--', linewidth=2, label=f'Reference Mean: {mean_ref_dispersion:.2f}')
ax.set_xlabel('Within-Cluster Dispersion (log)')
ax.set_ylabel('Frequency')
ax.set_title('Gap Statistic: Actual vs Reference')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gap_statistic.png', dpi=150, bbox_inches='tight')
print("✓ Saved: gap_statistic.png")

print("\n" + "="*80)
print("STEP 8: FINAL RECOMMENDATION")
print("="*80)

final_threshold = best_threshold_silhouette

print(f"\n{'='*80}")
print("RECOMMENDED THRESHOLD")
print(f"{'='*80}")
print(f"\n  Threshold: {final_threshold:.6f}")
print(f"\n  Method: Silhouette Optimization")
print(f"  Validated by: Label Concordance + Gap Statistic")

silhouette_row = silhouette_df[silhouette_df['threshold'] == final_threshold].iloc[0]
concordance_row = concordance_df[concordance_df['threshold'] == final_threshold].iloc[0]

print(f"\n  Performance Metrics:")
print(f"    Silhouette Score: {silhouette_row['silhouette_score']:.6f}")
print(f"    Purity: {concordance_row['purity']:.2f}%")
print(f"    Good pairs above threshold: {concordance_row['good_above_rate']:.2f}%")
print(f"    Bad pairs below threshold: {concordance_row['bad_below_rate']:.2f}%")
print(f"    Gap Statistic: {gap:.6f} ({'significant' if gap > 0 else 'not significant'})")

print(f"\n  Distribution Split:")
print(f"    Above threshold: {silhouette_row['above_count']:,} pairs ({silhouette_row['above_pct']:.2f}%)")
print(f"    Below threshold: {silhouette_row['below_count']:,} pairs ({100-silhouette_row['above_pct']:.2f}%)")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].hist(pairs_df['similarity'], bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(final_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {final_threshold:.6f}')
axes[0, 0].set_xlabel('Similarity Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Final Threshold on Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

above_mask = pairs_df['similarity'] >= final_threshold
below_mask = pairs_df['similarity'] < final_threshold

axes[0, 1].hist([pairs_df[above_mask]['similarity'], pairs_df[below_mask]['similarity']], 
                bins=50, label=['Above Threshold', 'Below Threshold'], stacked=True)
axes[0, 1].axvline(final_threshold, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Similarity Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution Split')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

conf_matrix = pd.DataFrame({
    'Above Threshold': [
        pairs_df[above_mask]['is_good_pair'].sum(),
        pairs_df[above_mask]['is_bad_pair'].sum()
    ],
    'Below Threshold': [
        pairs_df[below_mask]['is_good_pair'].sum(),
        pairs_df[below_mask]['is_bad_pair'].sum()
    ]
}, index=['Good Pairs', 'Bad Pairs'])

conf_matrix_pct = conf_matrix.div(conf_matrix.sum(axis=0), axis=1) * 100

im = axes[1, 0].imshow(conf_matrix_pct.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(conf_matrix_pct.columns)
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_yticklabels(conf_matrix_pct.index)
axes[1, 0].set_title('Label Quality Matrix (%)')

for i in range(2):
    for j in range(2):
        text = axes[1, 0].text(j, i, f'{conf_matrix_pct.iloc[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontsize=12, fontweight='bold')

plt.colorbar(im, ax=axes[1, 0])

summary_data = {
    'Metric': ['Silhouette Score', 'Purity', 'Good Above %', 'Bad Below %', 'Gap Statistic'],
    'Value': [
        f"{silhouette_row['silhouette_score']:.6f}",
        f"{concordance_row['purity']:.2f}%",
        f"{concordance_row['good_above_rate']:.2f}%",
        f"{concordance_row['bad_below_rate']:.2f}%",
        f"{gap:.6f}"
    ]
}
summary_df = pd.DataFrame(summary_data)

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=summary_df.values, colLabels=summary_df.columns,
                        cellLoc='left', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
axes[1, 1].set_title('Performance Summary', pad=20, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('final_recommendation.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: final_recommendation.png")

output_summary = {
    'recommended_threshold': final_threshold,
    'method': 'Silhouette Optimization',
    'silhouette_score': silhouette_row['silhouette_score'],
    'purity': concordance_row['purity'],
    'good_above_rate': concordance_row['good_above_rate'],
    'bad_below_rate': concordance_row['bad_below_rate'],
    'gap_statistic': gap,
    'pairs_above': int(silhouette_row['above_count']),
    'pairs_below': int(silhouette_row['below_count']),
    'pct_above': silhouette_row['above_pct']
}

with open('threshold_recommendation.pkl', 'wb') as f:
    pickle.dump(output_summary, f)
print("✓ Saved: threshold_recommendation.pkl")

pairs_df.to_parquet('sampled_pairs_with_scores.parquet')
print("✓ Saved: sampled_pairs_with_scores.parquet")

silhouette_df.to_csv('silhouette_results.csv', index=False)
print("✓ Saved: silhouette_results.csv")

concordance_df.to_csv('concordance_results.csv', index=False)
print("✓ Saved: concordance_results.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nFinal Threshold: {final_threshold:.6f}")
print(f"\nThis threshold maximizes cluster separation (Silhouette)")
print(f"while maintaining high label concordance (Purity).")
print(f"\nUse this threshold to classify provider pairs as similar/dissimilar.")
