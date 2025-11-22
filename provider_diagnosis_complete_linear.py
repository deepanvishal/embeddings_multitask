"""
Complete linear diagnostic for provider embedding discrepancies
All analyses included, just written linearly for easy debugging
"""

import numpy as np
import pandas as pd
import pickle
from scipy.sparse import load_npz
from scipy.spatial.distance import cosine, pdist
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SET YOUR PROVIDER PINS HERE (as integers)
# ============================================================================
PIN_A = 1111111
PIN_B = 111122  
PIN_C = 22222
# ============================================================================

print("="*80)
print("PROVIDER EMBEDDING DISCREPANCY DIAGNOSTIC ANALYSIS - COMPLETE VERSION")
print("="*80)

# ============================================================================
# LOAD ALL DATA
# ============================================================================
print("\nLoading data...")
proc_matrix = load_npz('procedure_vectors.npz')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()
final_embeddings = np.load('me2vec_provider_embeddings.npy')

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

with open('specialty_code_mappings.pkl', 'rb') as f:
    specialty_mappings = pickle.load(f)

# Try to load optional files
try:
    provider_init_embeddings = np.load('provider_init_embeddings.npy')
    print("✓ Initial embeddings loaded")
    has_init_embeddings = True
except:
    provider_init_embeddings = None
    print("✗ No initial embeddings (add np.save after line 221 in me2vec)")
    has_init_embeddings = False

try:
    code_embeddings = np.load('code_embeddings_dict.npy', allow_pickle=True).item()
    print("✓ Code embeddings loaded")
    # Get embedding dimension
    CODE_EMBEDDING_DIM = next(iter(code_embeddings.values())).shape[0]
    print(f"  Detected embedding dimension: {CODE_EMBEDDING_DIM}")
    has_code_embeddings = True
except:
    code_embeddings = {}
    print("✗ No code embeddings (add np.save after line 193 in me2vec)")
    CODE_EMBEDDING_DIM = None
    has_code_embeddings = False

try:
    cooccurrence_matrix = load_npz('cooccurrence_matrix.npz').tocsr()
    print("✓ Co-occurrence matrix loaded")
    has_cooccurrence = True
except:
    cooccurrence_matrix = None
    print("✗ No co-occurrence matrix (add save_npz after line 112 in me2vec)")
    has_cooccurrence = False

# Get specialty codes and filter procedure matrix
all_specialty_codes = sorted(list(set().union(*specialty_mappings['code_indices'].values())))
proc_matrix_filtered = proc_matrix[:, all_specialty_codes].tocsr()
n_providers = proc_matrix_filtered.shape[0]
n_codes = len(all_specialty_codes)

print(f"\nData dimensions:")
print(f"  Total providers: {n_providers}")
print(f"  Total procedure codes: {n_codes}")

# Create mappings
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}
code_to_idx = {code: idx for idx, code in enumerate(all_specialty_codes)}

# ============================================================================
# VALIDATE PINS AND GET INDICES
# ============================================================================
print("\n" + "="*80)
print("VALIDATING PROVIDER PINS")
print("="*80)

if PIN_A not in pin_to_idx:
    print(f"ERROR: PIN_A ({PIN_A}) not found!")
    print(f"Available PINs example: {list(pin_to_idx.keys())[:10]}")
    exit(1)

if PIN_B not in pin_to_idx:
    print(f"ERROR: PIN_B ({PIN_B}) not found!")
    exit(1)

if PIN_C not in pin_to_idx:
    print(f"ERROR: PIN_C ({PIN_C}) not found!")
    exit(1)

provider_A_idx = pin_to_idx[PIN_A]
provider_B_idx = pin_to_idx[PIN_B]
provider_C_idx = pin_to_idx[PIN_C]

print(f"Provider A: PIN {PIN_A} → Index {provider_A_idx}")
print(f"Provider B: PIN {PIN_B} → Index {provider_B_idx}")
print(f"Provider C: PIN {PIN_C} → Index {provider_C_idx}")

# Get specialties
if PIN_A in pin_to_label:
    print(f"  A Specialty: {pin_to_label[PIN_A]}")
if PIN_B in pin_to_label:
    print(f"  B Specialty: {pin_to_label[PIN_B]}")
if PIN_C in pin_to_label:
    print(f"  C Specialty: {pin_to_label[PIN_C]}")

# ============================================================================
# EXTRACT PROCEDURES FOR EACH PROVIDER
# ============================================================================
print("\n" + "="*60)
print("PROVIDER PROFILES")
print("="*60)

# Get provider A procedures
A_row = proc_matrix_filtered[provider_A_idx]
A_procedures = {}
for idx, count in zip(A_row.indices, A_row.data):
    A_procedures[all_specialty_codes[idx]] = count

# Get provider B procedures  
B_row = proc_matrix_filtered[provider_B_idx]
B_procedures = {}
for idx, count in zip(B_row.indices, B_row.data):
    B_procedures[all_specialty_codes[idx]] = count

# Get provider C procedures
C_row = proc_matrix_filtered[provider_C_idx]
C_procedures = {}
for idx, count in zip(C_row.indices, C_row.data):
    C_procedures[all_specialty_codes[idx]] = count

print(f"Provider A: PIN {PIN_A} - {len(A_procedures)} procedures")
print(f"Provider B: PIN {PIN_B} - {len(B_procedures)} procedures")
print(f"Provider C: PIN {PIN_C} - {len(C_procedures)} procedures")

# Calculate overlaps
A_codes = set(A_procedures.keys())
B_codes = set(B_procedures.keys())
C_codes = set(C_procedures.keys())

AB_shared = A_codes & B_codes
AC_shared = A_codes & C_codes
BC_shared = B_codes & C_codes
ABC_shared = A_codes & B_codes & C_codes

AB_only = AB_shared - C_codes
AC_only = AC_shared - B_codes
BC_only = BC_shared - A_codes

print(f"\nOverlap Analysis:")
print(f"  A∩B: {len(AB_shared)} procedures")
print(f"  A∩C: {len(AC_shared)} procedures")
print(f"  B∩C: {len(BC_shared)} procedures")
print(f"  A∩B∩C: {len(ABC_shared)} procedures")
print(f"  A∩B only (not C): {len(AB_only)} procedures")
print(f"  A∩C only (not B): {len(AC_only)} procedures")

# ============================================================================
# CALCULATE ALL SIMILARITIES
# ============================================================================

# Initial embeddings (if available)
if has_init_embeddings:
    sim_AB_init = 1 - cosine(provider_init_embeddings[provider_A_idx], 
                             provider_init_embeddings[provider_B_idx])
    sim_AC_init = 1 - cosine(provider_init_embeddings[provider_A_idx], 
                             provider_init_embeddings[provider_C_idx])
    sim_BC_init = 1 - cosine(provider_init_embeddings[provider_B_idx], 
                             provider_init_embeddings[provider_C_idx])
    
    print(f"\nInitial Embedding Similarities (Word2Vec + weighted average):")
    print(f"  A-B: {sim_AB_init:.4f}")
    print(f"  A-C: {sim_AC_init:.4f}")
    print(f"  B-C: {sim_BC_init:.4f}")

# Final embeddings
sim_AB_final = 1 - cosine(final_embeddings[provider_A_idx], 
                          final_embeddings[provider_B_idx])
sim_AC_final = 1 - cosine(final_embeddings[provider_A_idx], 
                          final_embeddings[provider_C_idx])
sim_BC_final = 1 - cosine(final_embeddings[provider_B_idx], 
                          final_embeddings[provider_C_idx])

print(f"\nFinal Embedding Similarities (after GAT):")
print(f"  A-B: {sim_AB_final:.4f}")
print(f"  A-C: {sim_AC_final:.4f}")
print(f"  B-C: {sim_BC_final:.4f}")

# Show changes
if has_init_embeddings:
    print(f"\nSimilarity Changes (Final - Initial):")
    print(f"  A-B: {sim_AB_final - sim_AB_init:+.4f}")
    print(f"  A-C: {sim_AC_final - sim_AC_init:+.4f}")
    print(f"  B-C: {sim_BC_final - sim_BC_init:+.4f}")

# ============================================================================
# BUILD DETAILED PROCEDURE DATAFRAME
# ============================================================================
print("\n" + "="*60)
print("BUILDING DETAILED PROCEDURE ANALYSIS")
print("="*60)

# Create comprehensive procedure list
all_provider_codes = A_codes | B_codes | C_codes
procedure_data = []

for code in all_provider_codes:
    row_data = {
        'code': code,
        'A_count': A_procedures.get(code, 0),
        'B_count': B_procedures.get(code, 0),
        'C_count': C_procedures.get(code, 0),
        'in_AB': code in AB_shared,
        'in_AC': code in AC_shared,
        'in_BC': code in BC_shared,
        'in_ABC': code in ABC_shared,
        'in_AB_only': code in AB_only,
        'in_AC_only': code in AC_only,
        'in_BC_only': code in BC_only
    }
    
    # Add global frequency
    code_idx = all_specialty_codes.index(code)
    providers_with_code = (proc_matrix_filtered[:, code_idx] > 0).sum()
    row_data['global_frequency'] = providers_with_code
    row_data['global_freq_pct'] = providers_with_code / n_providers
    
    # Add embedding norm if available
    if has_code_embeddings and code in code_embeddings:
        row_data['embedding_norm'] = np.linalg.norm(code_embeddings[code])
    else:
        row_data['embedding_norm'] = np.nan
    
    # Add co-occurrence metrics if available
    if has_cooccurrence and code_idx < cooccurrence_matrix.shape[0]:
        row = cooccurrence_matrix.getrow(code_idx)
        row_data['node_degree'] = (row.data > 0).sum()
        row_data['node_strength'] = row.sum()
        row_data['co_occurrence_diversity'] = row_data['node_degree']  # Same as degree for our purposes
        
        # Clustering coefficient (simplified)
        neighbors = row.nonzero()[1]
        if len(neighbors) > 1:
            # Sample neighbors if too many
            sample_size = min(len(neighbors), 50)
            if len(neighbors) > 50:
                sampled = np.random.choice(neighbors, sample_size, replace=False)
            else:
                sampled = neighbors
            
            neighbor_connections = 0
            for i in range(len(sampled)):
                for j in range(i+1, len(sampled)):
                    if cooccurrence_matrix[sampled[i], sampled[j]] > 0:
                        neighbor_connections += 1
            
            possible = sample_size * (sample_size - 1) / 2
            row_data['clustering_coef'] = neighbor_connections / possible if possible > 0 else 0
        else:
            row_data['clustering_coef'] = 0
    else:
        row_data['node_degree'] = np.nan
        row_data['node_strength'] = np.nan
        row_data['co_occurrence_diversity'] = np.nan
        row_data['clustering_coef'] = np.nan
    
    procedure_data.append(row_data)

proc_df = pd.DataFrame(procedure_data)
print(f"Created dataframe with {len(proc_df)} procedures")

# ============================================================================
# ANALYZE EXCLUSIVE OVERLAPS
# ============================================================================
print("\n" + "="*60)
print("EXCLUSIVE OVERLAP ANALYSIS")
print("="*60)

# Get subsets
ab_only_df = proc_df[proc_df['in_AB_only']]
ac_only_df = proc_df[proc_df['in_AC_only']]
abc_all_df = proc_df[proc_df['in_ABC']]

# AB-only analysis
print(f"\n=== AB-only Procedures (A&B share, C doesn't have) ===")
print(f"Count: {len(ab_only_df)}")
if len(ab_only_df) > 0:
    print(f"Avg Global Frequency: {ab_only_df['global_freq_pct'].mean():.4f} ({ab_only_df['global_freq_pct'].mean()*100:.1f}% of providers)")
    if not ab_only_df['embedding_norm'].isna().all():
        print(f"Avg Embedding Norm: {ab_only_df['embedding_norm'].mean():.3f}")
    if not ab_only_df['co_occurrence_diversity'].isna().all():
        print(f"Avg Co-occurrence Diversity: {ab_only_df['co_occurrence_diversity'].mean():.1f}")
    if not ab_only_df['node_degree'].isna().all():
        print(f"Avg Node Degree: {ab_only_df['node_degree'].mean():.1f}")
    if not ab_only_df['clustering_coef'].isna().all():
        print(f"Avg Clustering Coefficient: {ab_only_df['clustering_coef'].mean():.3f}")

# AC-only analysis
print(f"\n=== AC-only Procedures (A&C share, B doesn't have) ===")
print(f"Count: {len(ac_only_df)}")
if len(ac_only_df) > 0:
    print(f"Avg Global Frequency: {ac_only_df['global_freq_pct'].mean():.4f} ({ac_only_df['global_freq_pct'].mean()*100:.1f}% of providers)")
    if not ac_only_df['embedding_norm'].isna().all():
        print(f"Avg Embedding Norm: {ac_only_df['embedding_norm'].mean():.3f}")
    if not ac_only_df['co_occurrence_diversity'].isna().all():
        print(f"Avg Co-occurrence Diversity: {ac_only_df['co_occurrence_diversity'].mean():.1f}")
    if not ac_only_df['node_degree'].isna().all():
        print(f"Avg Node Degree: {ac_only_df['node_degree'].mean():.1f}")
    if not ac_only_df['clustering_coef'].isna().all():
        print(f"Avg Clustering Coefficient: {ac_only_df['clustering_coef'].mean():.3f}")

# ABC-all analysis
print(f"\n=== ABC Procedures (All three share) ===")
print(f"Count: {len(abc_all_df)}")
if len(abc_all_df) > 0:
    print(f"Avg Global Frequency: {abc_all_df['global_freq_pct'].mean():.4f} ({abc_all_df['global_freq_pct'].mean()*100:.1f}% of providers)")
    if not abc_all_df['embedding_norm'].isna().all():
        print(f"Avg Embedding Norm: {abc_all_df['embedding_norm'].mean():.3f}")

# ============================================================================
# STATISTICAL TESTS
# ============================================================================
if len(ab_only_df) > 0 and len(ac_only_df) > 0:
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON (t-tests)")
    print("="*60)
    print("Metric                        t-stat    p-value   Significance")
    print("-" * 60)
    
    # Test each metric
    metrics_to_test = ['global_freq_pct', 'embedding_norm', 'co_occurrence_diversity', 
                      'node_degree', 'clustering_coef']
    
    for metric in metrics_to_test:
        if metric in ab_only_df.columns and metric in ac_only_df.columns:
            ab_values = ab_only_df[metric].dropna()
            ac_values = ac_only_df[metric].dropna()
            
            if len(ab_values) > 1 and len(ac_values) > 1:
                t_stat, p_value = stats.ttest_ind(ab_values, ac_values)
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"{metric:28s} {t_stat:8.3f} {p_value:9.4f}  {sig}")

# ============================================================================
# TOP PROCEDURES BY VARIOUS METRICS
# ============================================================================
print("\n" + "="*60)
print("TOP PROCEDURES")
print("="*60)

# Top AB-only by embedding norm
if len(ab_only_df) > 0 and not ab_only_df['embedding_norm'].isna().all():
    print("\n=== Top 5 AB-only by Embedding Norm ===")
    top_ab = ab_only_df.nlargest(min(5, len(ab_only_df)), 'embedding_norm')[
        ['code', 'A_count', 'B_count', 'global_freq_pct', 'embedding_norm']
    ]
    print(top_ab.to_string(index=False))

# Top AC-only by global frequency
if len(ac_only_df) > 0:
    print("\n=== Top 5 AC-only by Global Frequency ===")
    top_ac = ac_only_df.nlargest(min(5, len(ac_only_df)), 'global_freq_pct')[
        ['code', 'A_count', 'C_count', 'global_freq_pct', 'embedding_norm']
    ]
    print(top_ac.to_string(index=False))

# ============================================================================
# EMBEDDING CONTRIBUTION ANALYSIS
# ============================================================================
if has_code_embeddings and has_init_embeddings:
    print("\n" + "="*60)
    print("EMBEDDING CONTRIBUTION ANALYSIS")
    print("="*60)
    
    provider_A_emb = provider_init_embeddings[provider_A_idx]
    
    # AB-only contribution
    if len(AB_only) > 0:
        ab_contribution = np.zeros(CODE_EMBEDDING_DIM, dtype=np.float32)
        ab_weight = 0
        missing = 0
        for code in AB_only:
            if code in code_embeddings:
                count = A_procedures.get(code, 0)
                if count > 0:
                    ab_contribution += code_embeddings[code] * count
                    ab_weight += count
            else:
                missing += 1
        
        if ab_weight > 0:
            ab_contribution /= ab_weight
            sim_ab = 1 - cosine(ab_contribution, provider_A_emb)
            print(f"\nProvider A procedure contributions:")
            print(f"  AB-only contribution alignment: {sim_ab:.3f}")
            if missing > 0:
                print(f"  Warning: {missing}/{len(AB_only)} codes missing embeddings")
    
    # AC-only contribution
    if len(AC_only) > 0:
        ac_contribution = np.zeros(CODE_EMBEDDING_DIM, dtype=np.float32)
        ac_weight = 0
        missing = 0
        for code in AC_only:
            if code in code_embeddings:
                count = A_procedures.get(code, 0)
                if count > 0:
                    ac_contribution += code_embeddings[code] * count
                    ac_weight += count
            else:
                missing += 1
        
        if ac_weight > 0:
            ac_contribution /= ac_weight
            sim_ac = 1 - cosine(ac_contribution, provider_A_emb)
            print(f"  AC-only contribution alignment: {sim_ac:.3f}")
            if missing > 0:
                print(f"  Warning: {missing}/{len(AC_only)} codes missing embeddings")
    
    # ABC-all contribution
    if len(ABC_shared) > 0:
        abc_contribution = np.zeros(CODE_EMBEDDING_DIM, dtype=np.float32)
        abc_weight = 0
        missing = 0
        for code in ABC_shared:
            if code in code_embeddings:
                count = A_procedures.get(code, 0)
                if count > 0:
                    abc_contribution += code_embeddings[code] * count
                    abc_weight += count
            else:
                missing += 1
        
        if abc_weight > 0:
            abc_contribution /= abc_weight
            sim_abc = 1 - cosine(abc_contribution, provider_A_emb)
            print(f"  ABC-all contribution alignment: {sim_abc:.3f}")
    
    # Compare contributions
    if len(AB_only) > 0 and len(AC_only) > 0 and ab_weight > 0 and ac_weight > 0:
        sim = 1 - cosine(ab_contribution, ac_contribution)
        print(f"\nInter-contribution similarities:")
        print(f"  AB vs AC contributions: {sim:.3f}")

# ============================================================================
# EMBEDDING SPACE GEOMETRY ANALYSIS
# ============================================================================
if has_code_embeddings and len(ab_only_df) > 0 and len(ac_only_df) > 0:
    print("\n" + "="*60)
    print("EMBEDDING SPACE GEOMETRY ANALYSIS")
    print("="*60)
    
    # AB-only geometry
    ab_codes_with_emb = [code for code in AB_only if code in code_embeddings]
    if len(ab_codes_with_emb) >= 2:
        ab_embs = np.array([code_embeddings[code] for code in ab_codes_with_emb])
        
        # Internal coherence
        ab_distances = pdist(ab_embs, metric='cosine')
        ab_coherence = 1 - ab_distances.mean() if len(ab_distances) > 0 else 0
        
        # Centroid
        ab_centroid = ab_embs.mean(axis=0)
        ab_centroid_norm = np.linalg.norm(ab_centroid)
        
        # Spread
        ab_distances_to_centroid = [cosine(emb, ab_centroid) for emb in ab_embs]
        ab_spread = np.std(ab_distances_to_centroid)
        
        # PCA dimensionality
        try:
            pca_ab = PCA()
            pca_ab.fit(ab_embs)
            cumsum_ab = np.cumsum(pca_ab.explained_variance_ratio_)
            dims_90_ab = np.argmax(cumsum_ab >= 0.9) + 1 if np.any(cumsum_ab >= 0.9) else len(cumsum_ab)
        except:
            dims_90_ab = -1
        
        print(f"\nAB-only ({len(ab_codes_with_emb)} procedures):")
        print(f"  Internal Coherence: {ab_coherence:.3f} (1=identical, 0=orthogonal)")
        print(f"  Centroid Norm: {ab_centroid_norm:.3f}")
        print(f"  Spread from Centroid: {ab_spread:.3f}")
        if dims_90_ab > 0:
            print(f"  Effective Dimensions (90% var): {dims_90_ab}/{CODE_EMBEDDING_DIM}")
    
    # AC-only geometry
    ac_codes_with_emb = [code for code in AC_only if code in code_embeddings]
    if len(ac_codes_with_emb) >= 2:
        ac_embs = np.array([code_embeddings[code] for code in ac_codes_with_emb])
        
        # Internal coherence
        ac_distances = pdist(ac_embs, metric='cosine')
        ac_coherence = 1 - ac_distances.mean() if len(ac_distances) > 0 else 0
        
        # Centroid
        ac_centroid = ac_embs.mean(axis=0)
        ac_centroid_norm = np.linalg.norm(ac_centroid)
        
        # Spread
        ac_distances_to_centroid = [cosine(emb, ac_centroid) for emb in ac_embs]
        ac_spread = np.std(ac_distances_to_centroid)
        
        # PCA dimensionality
        try:
            pca_ac = PCA()
            pca_ac.fit(ac_embs)
            cumsum_ac = np.cumsum(pca_ac.explained_variance_ratio_)
            dims_90_ac = np.argmax(cumsum_ac >= 0.9) + 1 if np.any(cumsum_ac >= 0.9) else len(cumsum_ac)
        except:
            dims_90_ac = -1
        
        print(f"\nAC-only ({len(ac_codes_with_emb)} procedures):")
        print(f"  Internal Coherence: {ac_coherence:.3f} (1=identical, 0=orthogonal)")
        print(f"  Centroid Norm: {ac_centroid_norm:.3f}")
        print(f"  Spread from Centroid: {ac_spread:.3f}")
        if dims_90_ac > 0:
            print(f"  Effective Dimensions (90% var): {dims_90_ac}/{CODE_EMBEDDING_DIM}")
    
    # Compare centroids if both exist
    if len(ab_codes_with_emb) >= 2 and len(ac_codes_with_emb) >= 2:
        centroid_sim = 1 - cosine(ab_centroid, ac_centroid)
        print(f"\nCentroid Similarity AB vs AC: {centroid_sim:.3f}")

# ============================================================================
# FINAL DIAGNOSIS
# ============================================================================
print("\n" + "="*60)
print("FINAL DIAGNOSIS")
print("="*60)

# Hypothesis testing results
print("\nHypothesis Testing Results:")
if len(ab_only_df) > 0 and len(ac_only_df) > 0:
    if ab_only_df['global_freq_pct'].mean() < ac_only_df['global_freq_pct'].mean():
        print("  ✓ AB procedures are RARER (more distinctive)")
    else:
        print("  ✗ AB procedures are NOT rarer")
    
    if not ab_only_df['embedding_norm'].isna().all() and not ac_only_df['embedding_norm'].isna().all():
        if ab_only_df['embedding_norm'].mean() > ac_only_df['embedding_norm'].mean():
            print("  ✓ AB procedures have STRONGER embeddings")
        else:
            print("  ✗ AB procedures do NOT have stronger embeddings")
    
    if not ab_only_df['clustering_coef'].isna().all() and not ac_only_df['clustering_coef'].isna().all():
        if ab_only_df['clustering_coef'].mean() > ac_only_df['clustering_coef'].mean():
            print("  ✓ AB procedures form TIGHTER CLUSTERS")
        else:
            print("  ✗ AB procedures do NOT form tighter clusters")
    
    if not ac_only_df['co_occurrence_diversity'].isna().all() and not ab_only_df['co_occurrence_diversity'].isna().all():
        if ac_only_df['co_occurrence_diversity'].mean() > ab_only_df['co_occurrence_diversity'].mean():
            print("  ✓ AC procedures are more GENERIC (higher co-occurrence diversity)")
        else:
            print("  ✗ AC procedures are NOT more generic")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"Procedure Overlap:")
print(f"  A-B: {len(AB_shared)} procedures")
print(f"  A-C: {len(AC_shared)} procedures")

print(f"\nFinal Embedding Similarity:")
print(f"  A-B: {sim_AB_final:.4f}")
print(f"  A-C: {sim_AC_final:.4f}")

if has_init_embeddings:
    print(f"\nInitial Embedding Similarity:")
    print(f"  A-B: {sim_AB_init:.4f}")
    print(f"  A-C: {sim_AC_init:.4f}")

# ============================================================================
# DISCREPANCY ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("DISCREPANCY ANALYSIS")
print("="*60)

# Check for discrepancy
if len(AB_shared) > len(AC_shared) and sim_AC_final > sim_AB_final:
    print("✓ DISCREPANCY FOUND: Provider C is more similar to A than B,")
    print(f"  despite B having {len(AB_shared) - len(AC_shared)} more shared procedures.")
    
    if has_init_embeddings:
        if sim_AC_init > sim_AB_init:
            print("\n→ Root Cause: WORD2VEC/NODE2VEC STAGE")
            print("  The discrepancy is already present in initial embeddings.")
            print("  This stems from procedure co-occurrence patterns in the graph.")
        else:
            print("\n→ Root Cause: GAT/FINE-TUNING STAGE")
            print("  Initial embeddings had AB ≥ AC, but GAT flipped it.")
            print("  The attention mechanism learned to weight procedures differently.")

elif len(AC_shared) > len(AB_shared) and sim_AB_final > sim_AC_final:
    print("✓ DISCREPANCY FOUND: Provider B is more similar to A than C,")
    print(f"  despite C having {len(AC_shared) - len(AB_shared)} more shared procedures.")
    
    if has_init_embeddings:
        if sim_AB_init > sim_AC_init:
            print("\n→ Root Cause: WORD2VEC/NODE2VEC STAGE")
            print("  The discrepancy is already present in initial embeddings.")
            print("  This stems from procedure co-occurrence patterns in the graph.")
        else:
            print("\n→ Root Cause: GAT/FINE-TUNING STAGE")
            print("  Initial embeddings had AC ≥ AB, but GAT flipped it.")
            print("  The attention mechanism learned to weight procedures differently.")

else:
    print("✓ NO DISCREPANCY: Overlap and embedding similarity are directionally consistent.")
    if len(AB_shared) > len(AC_shared):
        print(f"  B has more overlap AND higher similarity with A (as expected).")
    else:
        print(f"  C has more overlap AND higher similarity with A (as expected).")

print(f"\nConclusion:")
print("The embedding model prioritizes procedure co-occurrence patterns and")
print("learned attention weights over raw procedure overlap counts.")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save detailed procedure dataframe
detailed_filename = f"procedure_details_{PIN_A}_{PIN_B}_{PIN_C}.csv"
proc_df.to_csv(detailed_filename, index=False)
print(f"Detailed procedure analysis saved to: {detailed_filename}")

# Save summary results
summary_data = {
    'Provider_A_PIN': [PIN_A],
    'Provider_B_PIN': [PIN_B],
    'Provider_C_PIN': [PIN_C],
    'A_procedures': [len(A_procedures)],
    'B_procedures': [len(B_procedures)],
    'C_procedures': [len(C_procedures)],
    'AB_shared': [len(AB_shared)],
    'AC_shared': [len(AC_shared)],
    'BC_shared': [len(BC_shared)],
    'AB_only': [len(AB_only)],
    'AC_only': [len(AC_only)],
    'sim_AB_final': [sim_AB_final],
    'sim_AC_final': [sim_AC_final],
    'sim_BC_final': [sim_BC_final]
}

if has_init_embeddings:
    summary_data['sim_AB_init'] = [sim_AB_init]
    summary_data['sim_AC_init'] = [sim_AC_init]
    summary_data['sim_BC_init'] = [sim_BC_init]
    summary_data['change_AB'] = [sim_AB_final - sim_AB_init]
    summary_data['change_AC'] = [sim_AC_final - sim_AC_init]

if len(ab_only_df) > 0:
    summary_data['AB_only_avg_freq'] = [ab_only_df['global_freq_pct'].mean()]
    if not ab_only_df['embedding_norm'].isna().all():
        summary_data['AB_only_avg_norm'] = [ab_only_df['embedding_norm'].mean()]

if len(ac_only_df) > 0:
    summary_data['AC_only_avg_freq'] = [ac_only_df['global_freq_pct'].mean()]
    if not ac_only_df['embedding_norm'].isna().all():
        summary_data['AC_only_avg_norm'] = [ac_only_df['embedding_norm'].mean()]

summary_df = pd.DataFrame(summary_data)
summary_filename = f"summary_{PIN_A}_{PIN_B}_{PIN_C}.csv"
summary_df.to_csv(summary_filename, index=False)
print(f"Summary results saved to: {summary_filename}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
