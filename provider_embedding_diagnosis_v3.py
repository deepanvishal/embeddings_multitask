"""
Provider Embedding Discrepancy Diagnostic Analysis
Analyzes why embedding similarities don't match procedure overlap patterns
Usage: python provider_embedding_diagnosis_v3.py
"""

import numpy as np
import pandas as pd
import pickle
from scipy.sparse import load_npz, save_npz
from scipy.spatial.distance import cosine, pdist
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURE YOUR PROVIDERS HERE
# ============================================================================
PIN_A = 1111111  # Provider A PIN
PIN_B = 111122   # Provider B PIN  
PIN_C = 22222    # Provider C PIN
# ============================================================================

print("="*80)
print("PROVIDER EMBEDDING DISCREPANCY DIAGNOSTIC ANALYSIS")
print("="*80)

# Load all necessary data
print("\nLoading data...")
proc_matrix = load_npz('procedure_vectors.npz')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

with open('specialty_code_mappings.pkl', 'rb') as f:
    specialty_mappings = pickle.load(f)

specialty_code_indices = specialty_mappings['code_indices']

# Load embedding-related data
print("Loading embeddings...")
final_embeddings = np.load('me2vec_provider_embeddings.npy')

# Check if these files exist, if not, provide instructions
import os
if not os.path.exists('provider_init_embeddings.npy'):
    print("\nWARNING: provider_init_embeddings.npy not found!")
    print("Please add this line after line 221 in your me2vec code:")
    print("np.save('provider_init_embeddings.npy', provider_init_embeddings)")
    provider_init_embeddings = None
else:
    provider_init_embeddings = np.load('provider_init_embeddings.npy')

if not os.path.exists('code_embeddings_dict.npy'):
    print("\nWARNING: code_embeddings_dict.npy not found!")
    print("Please add this line after line 193 in your me2vec code:")
    print("np.save('code_embeddings_dict.npy', code_embeddings)")
    code_embeddings = {}
else:
    code_embeddings = np.load('code_embeddings_dict.npy', allow_pickle=True).item()

if not os.path.exists('cooccurrence_matrix.npz'):
    print("\nWARNING: cooccurrence_matrix.npz not found!")
    print("Please add this line after line 112 in your me2vec code:")
    print("from scipy.sparse import save_npz")
    print("save_npz('cooccurrence_matrix.npz', cooccurrence_matrix)")
    cooccurrence_matrix = None
else:
    cooccurrence_matrix = load_npz('cooccurrence_matrix.npz')

# Get specialty codes
all_specialty_codes = sorted(list(set().union(*specialty_code_indices.values())))
proc_matrix_filtered = proc_matrix[:, all_specialty_codes]

# Create mappings
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}
code_to_idx = {code: idx for idx, code in enumerate(all_specialty_codes)}

# Get embedding dimension dynamically
if code_embeddings:
    CODE_EMBEDDING_DIM = next(iter(code_embeddings.values())).shape[0]
    print(f"Detected code embedding dimension: {CODE_EMBEDDING_DIM}")
else:
    CODE_EMBEDDING_DIM = None
    print("Code embeddings not available, dimension unknown")

# Validate PINs and get indices
print("\n" + "="*80)
print("VALIDATING PROVIDER PINS")
print("="*80)

if PIN_A not in pin_to_idx:
    print(f"ERROR: PIN_A ({PIN_A}) not found in data!")
    print(f"Available PINs example: {list(pin_to_idx.keys())[:10]}")
    exit(1)

if PIN_B not in pin_to_idx:
    print(f"ERROR: PIN_B ({PIN_B}) not found in data!")
    exit(1)

if PIN_C not in pin_to_idx:
    print(f"ERROR: PIN_C ({PIN_C}) not found in data!")
    exit(1)

provider_A_idx = pin_to_idx[PIN_A]
provider_B_idx = pin_to_idx[PIN_B]
provider_C_idx = pin_to_idx[PIN_C]

print(f"Provider A: PIN {PIN_A} → Index {provider_A_idx}")
print(f"Provider B: PIN {PIN_B} → Index {provider_B_idx}")
print(f"Provider C: PIN {PIN_C} → Index {provider_C_idx}")

def analyze_provider_procedures(provider_A_idx, provider_B_idx, provider_C_idx):
    """
    Comprehensive analysis of procedure patterns for three providers
    """
    
    # Get procedure indices and counts for each provider
    A_procs = proc_matrix_filtered[provider_A_idx]
    B_procs = proc_matrix_filtered[provider_B_idx]
    C_procs = proc_matrix_filtered[provider_C_idx]
    
    # Convert to dictionaries {procedure_code: claim_count}
    A_dict = {all_specialty_codes[idx]: count for idx, count in zip(A_procs.indices, A_procs.data)}
    B_dict = {all_specialty_codes[idx]: count for idx, count in zip(B_procs.indices, B_procs.data)}
    C_dict = {all_specialty_codes[idx]: count for idx, count in zip(C_procs.indices, C_procs.data)}
    
    # Find overlaps
    all_codes = set(A_dict.keys()) | set(B_dict.keys()) | set(C_dict.keys())
    
    results = []
    for code in all_codes:
        results.append({
            'code': code,
            'A_count': A_dict.get(code, 0),
            'B_count': B_dict.get(code, 0),
            'C_count': C_dict.get(code, 0),
            'in_AB': code in A_dict and code in B_dict,
            'in_AC': code in A_dict and code in C_dict,
            'in_BC': code in B_dict and code in C_dict,
            'in_ABC': code in A_dict and code in B_dict and code in C_dict
        })
    
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print("PROVIDER PROFILES")
    print(f"{'='*60}")
    print(f"Provider A: PIN {PIN_A} - {len(A_dict)} procedures")
    print(f"Provider B: PIN {PIN_B} - {len(B_dict)} procedures")
    print(f"Provider C: PIN {PIN_C} - {len(C_dict)} procedures")
    
    # Get specialties if available
    if PIN_A in pin_to_label:
        print(f"  A Specialty: {pin_to_label[PIN_A]}")
    if PIN_B in pin_to_label:
        print(f"  B Specialty: {pin_to_label[PIN_B]}")
    if PIN_C in pin_to_label:
        print(f"  C Specialty: {pin_to_label[PIN_C]}")
    
    print(f"\nOverlap Analysis:")
    print(f"  A∩B: {df['in_AB'].sum()} procedures")
    print(f"  A∩C: {df['in_AC'].sum()} procedures")
    print(f"  B∩C: {df['in_BC'].sum()} procedures")
    print(f"  A∩B∩C: {df['in_ABC'].sum()} procedures")
    print(f"  A∩B only (not C): {(df['in_AB'] & ~df['in_AC']).sum()} procedures")
    print(f"  A∩C only (not B): {(df['in_AC'] & ~df['in_AB']).sum()} procedures")
    
    # Calculate INITIAL embedding similarities (if available)
    if provider_init_embeddings is not None:
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
    else:
        sim_AB_init = sim_AC_init = sim_BC_init = None
    
    # Calculate FINAL embedding similarities
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
    
    # Show changes if initial embeddings available
    if provider_init_embeddings is not None:
        print(f"\nSimilarity Changes (Final - Initial):")
        print(f"  A-B: {sim_AB_final - sim_AB_init:+.4f}")
        print(f"  A-C: {sim_AC_final - sim_AC_init:+.4f}")
        print(f"  B-C: {sim_BC_final - sim_BC_init:+.4f}")
    
    return df

def analyze_procedure_meaningfulness(proc_df):
    """
    Determine what makes a procedure 'meaningful' in the embedding space
    """
    
    # Global statistics
    n_providers = proc_matrix_filtered.shape[0]
    
    # Initialize new columns
    proc_df['global_frequency'] = 0
    proc_df['global_freq_pct'] = 0.0
    proc_df['embedding_norm'] = 0.0
    proc_df['co_occurrence_diversity'] = 0
    
    for idx, row in proc_df.iterrows():
        code = row['code']
        if code not in code_to_idx:
            continue
            
        code_idx = code_to_idx[code]
        
        # 1. FREQUENCY: How common is this procedure globally?
        providers_with_code = (proc_matrix_filtered[:, code_idx] > 0).sum()
        proc_df.loc[idx, 'global_frequency'] = providers_with_code
        proc_df.loc[idx, 'global_freq_pct'] = providers_with_code / n_providers
        
        # 2. EMBEDDING NORM: Procedures with high norms are more "distinctive"
        if code in code_embeddings:
            proc_df.loc[idx, 'embedding_norm'] = np.linalg.norm(code_embeddings[code])
        
        # 3. CO-OCCURRENCE DIVERSITY: How many different procedures does this co-occur with?
        if cooccurrence_matrix is not None and code_idx < cooccurrence_matrix.shape[0]:
            co_occurring = (cooccurrence_matrix[code_idx, :] > 0).sum()
            proc_df.loc[idx, 'co_occurrence_diversity'] = co_occurring
    
    return proc_df

def analyze_graph_positions(proc_df):
    """
    Where do these procedures sit in the co-occurrence graph?
    """
    
    if cooccurrence_matrix is None:
        print("Skipping graph position analysis (cooccurrence_matrix not available)")
        return proc_df
    
    # Initialize new columns
    proc_df['node_degree'] = 0
    proc_df['node_strength'] = 0.0
    proc_df['clustering_coef'] = 0.0
    
    for idx, row in proc_df.iterrows():
        code = row['code']
        if code not in code_to_idx:
            continue
        
        code_idx = code_to_idx[code]
        
        if code_idx >= cooccurrence_matrix.shape[0]:
            continue
            
        # Node degree (how many other procedures it co-occurs with)
        row_data = cooccurrence_matrix.getrow(code_idx)
        proc_df.loc[idx, 'node_degree'] = (row_data > 0).sum()
        
        # Node strength (sum of co-occurrence weights)
        proc_df.loc[idx, 'node_strength'] = row_data.sum()
        
        # Clustering coefficient (simplified for performance)
        neighbors = row_data.nonzero()[1]
        if len(neighbors) > 1:
            neighbor_connections = 0
            sample_size = min(len(neighbors), 50)
            sampled_neighbors = np.random.choice(neighbors, sample_size, replace=False) if len(neighbors) > 50 else neighbors
            
            for i in range(len(sampled_neighbors)):
                for j in range(i+1, len(sampled_neighbors)):
                    if cooccurrence_matrix[sampled_neighbors[i], sampled_neighbors[j]] > 0:
                        neighbor_connections += 1
            
            possible_connections = sample_size * (sample_size - 1) / 2
            proc_df.loc[idx, 'clustering_coef'] = neighbor_connections / possible_connections if possible_connections > 0 else 0
        else:
            proc_df.loc[idx, 'clustering_coef'] = 0
    
    return proc_df

def compare_exclusive_overlaps(proc_df):
    """
    Deep dive into procedures that A shares with B but not C, and vice versa
    """
    
    # AB-only procedures (shared by A&B but not C)
    ab_only = proc_df[(proc_df['in_AB']) & (~proc_df['in_AC'])]
    
    # AC-only procedures (shared by A&C but not B)
    ac_only = proc_df[(proc_df['in_AC']) & (~proc_df['in_AB'])]
    
    # ABC procedures (shared by all)
    abc_all = proc_df[proc_df['in_ABC']]
    
    print(f"\n{'='*60}")
    print("EXCLUSIVE OVERLAP ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\n=== AB-only Procedures (A&B share, C doesn't have) ===")
    print(f"Count: {len(ab_only)}")
    if len(ab_only) > 0:
        print(f"Avg Global Frequency: {ab_only['global_freq_pct'].mean():.4f} ({ab_only['global_freq_pct'].mean()*100:.1f}% of providers)")
        print(f"Avg Embedding Norm: {ab_only['embedding_norm'].mean():.3f}")
        print(f"Avg Co-occurrence Diversity: {ab_only['co_occurrence_diversity'].mean():.1f}")
        if 'node_degree' in ab_only.columns:
            print(f"Avg Node Degree: {ab_only['node_degree'].mean():.1f}")
            print(f"Avg Clustering Coefficient: {ab_only['clustering_coef'].mean():.3f}")
    
    print(f"\n=== AC-only Procedures (A&C share, B doesn't have) ===")
    print(f"Count: {len(ac_only)}")
    if len(ac_only) > 0:
        print(f"Avg Global Frequency: {ac_only['global_freq_pct'].mean():.4f} ({ac_only['global_freq_pct'].mean()*100:.1f}% of providers)")
        print(f"Avg Embedding Norm: {ac_only['embedding_norm'].mean():.3f}")
        print(f"Avg Co-occurrence Diversity: {ac_only['co_occurrence_diversity'].mean():.1f}")
        if 'node_degree' in ac_only.columns:
            print(f"Avg Node Degree: {ac_only['node_degree'].mean():.1f}")
            print(f"Avg Clustering Coefficient: {ac_only['clustering_coef'].mean():.3f}")
    
    print(f"\n=== ABC Procedures (All three share) ===")
    print(f"Count: {len(abc_all)}")
    if len(abc_all) > 0:
        print(f"Avg Global Frequency: {abc_all['global_freq_pct'].mean():.4f} ({abc_all['global_freq_pct'].mean()*100:.1f}% of providers)")
        print(f"Avg Embedding Norm: {abc_all['embedding_norm'].mean():.3f}")
    
    # Statistical tests
    if len(ab_only) > 0 and len(ac_only) > 0:
        print(f"\n{'='*60}")
        print("STATISTICAL COMPARISON (t-tests)")
        print(f"{'='*60}")
        print("Metric                        t-stat    p-value   Significance")
        print("-" * 60)
        
        metrics_to_test = ['global_freq_pct', 'embedding_norm', 'co_occurrence_diversity']
        if 'node_degree' in proc_df.columns:
            metrics_to_test.extend(['node_degree', 'clustering_coef'])
        
        for metric in metrics_to_test:
            if metric in ab_only.columns and metric in ac_only.columns:
                ab_values = ab_only[metric].dropna()
                ac_values = ac_only[metric].dropna()
                
                if len(ab_values) > 1 and len(ac_values) > 1:
                    t_stat, p_value = stats.ttest_ind(ab_values, ac_values)
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"{metric:28s} {t_stat:8.3f} {p_value:9.4f}  {sig}")
    
    # Show top examples
    if len(ab_only) > 0 and 'embedding_norm' in ab_only.columns:
        print(f"\n=== Top 5 AB-only by Embedding Norm ===")
        top_ab = ab_only.nlargest(min(5, len(ab_only)), 'embedding_norm')[['code', 'A_count', 'B_count', 'global_freq_pct', 'embedding_norm']]
        print(top_ab.to_string(index=False))
    
    if len(ac_only) > 0:
        print(f"\n=== Top 5 AC-only by Global Frequency ===")
        top_ac = ac_only.nlargest(min(5, len(ac_only)), 'global_freq_pct')[['code', 'A_count', 'C_count', 'global_freq_pct', 'embedding_norm']]
        print(top_ac.to_string(index=False))
    
    return ab_only, ac_only, abc_all

def analyze_embedding_contributions(provider_A_idx, provider_B_idx, provider_C_idx, proc_df):
    """
    How much does each procedure set contribute to the final embedding?
    """
    
    if provider_init_embeddings is None or len(code_embeddings) == 0:
        print("\nSkipping embedding contribution analysis (embeddings not available)")
        return
    
    print(f"\n{'='*60}")
    print("EMBEDDING CONTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    # Provider embeddings
    provider_A_emb = provider_init_embeddings[provider_A_idx]
    provider_B_emb = provider_init_embeddings[provider_B_idx]
    provider_C_emb = provider_init_embeddings[provider_C_idx]
    
    # Calculate contribution of different procedure sets
    ab_only_codes = proc_df[(proc_df['in_AB']) & (~proc_df['in_AC'])]['code'].values
    ac_only_codes = proc_df[(proc_df['in_AC']) & (~proc_df['in_AB'])]['code'].values
    bc_only_codes = proc_df[(proc_df['in_BC']) & (~proc_df['in_AB']) & (~proc_df['in_AC'])]['code'].values
    abc_all_codes = proc_df[proc_df['in_ABC']]['code'].values
    
    def get_weighted_embedding(codes, provider_counts, provider_name):
        if CODE_EMBEDDING_DIM is None:
            print(f"  Warning: Cannot compute embeddings (dimension unknown)")
            return None
            
        emb_sum = np.zeros(CODE_EMBEDDING_DIM, dtype=np.float32)
        weight_sum = 0
        missing_codes = 0
        
        for code in codes:
            if code in code_embeddings:
                count = proc_df[proc_df['code'] == code][f'{provider_name}_count'].values[0]
                if count > 0:
                    emb_sum += code_embeddings[code] * count
                    weight_sum += count
            else:
                missing_codes += 1
        
        if missing_codes > 0:
            print(f"  Warning: {missing_codes}/{len(codes)} codes missing embeddings for {provider_name}")
        
        return emb_sum / weight_sum if weight_sum > 0 else emb_sum
    
    # Get contributions for provider A
    print("\nProvider A procedure contributions:")
    contribution_results = {}
    
    if len(ab_only_codes) > 0:
        ab_contribution_A = get_weighted_embedding(ab_only_codes, proc_df, 'A')
        if ab_contribution_A is not None and np.any(ab_contribution_A) and np.any(provider_A_emb):
            sim_ab = 1 - cosine(ab_contribution_A, provider_A_emb)
            print(f"  AB-only contribution alignment: {sim_ab:.3f}")
            contribution_results['ab'] = ab_contribution_A
    
    if len(ac_only_codes) > 0:
        ac_contribution_A = get_weighted_embedding(ac_only_codes, proc_df, 'A')
        if ac_contribution_A is not None and np.any(ac_contribution_A) and np.any(provider_A_emb):
            sim_ac = 1 - cosine(ac_contribution_A, provider_A_emb)
            print(f"  AC-only contribution alignment: {sim_ac:.3f}")
            contribution_results['ac'] = ac_contribution_A
    
    if len(abc_all_codes) > 0:
        abc_contribution_A = get_weighted_embedding(abc_all_codes, proc_df, 'A')
        if abc_contribution_A is not None and np.any(abc_contribution_A) and np.any(provider_A_emb):
            sim_abc = 1 - cosine(abc_contribution_A, provider_A_emb)
            print(f"  ABC-all contribution alignment: {sim_abc:.3f}")
            contribution_results['abc'] = abc_contribution_A
    
    # Compare contribution vectors
    if 'ab' in contribution_results and 'ac' in contribution_results:
        print("\nInter-contribution similarities:")
        if np.any(contribution_results['ab']) and np.any(contribution_results['ac']):
            sim = 1 - cosine(contribution_results['ab'], contribution_results['ac'])
            print(f"  AB vs AC contributions: {sim:.3f}")

def analyze_embedding_space_geometry(ab_only, ac_only, abc_all):
    """
    Analyze how procedure sets cluster in embedding space
    """
    
    if len(code_embeddings) == 0:
        print("\nSkipping embedding space geometry analysis (embeddings not available)")
        return
    
    print(f"\n{'='*60}")
    print("EMBEDDING SPACE GEOMETRY ANALYSIS")
    print(f"{'='*60}")
    
    def analyze_set_geometry(proc_set, set_name):
        codes_with_emb = [code for code in proc_set['code'].values if code in code_embeddings]
        
        if len(codes_with_emb) < 2:
            print(f"\n{set_name}: Too few procedures with embeddings ({len(codes_with_emb)})")
            return None
        
        embs = np.array([code_embeddings[code] for code in codes_with_emb])
        
        # Internal coherence
        distances = pdist(embs, metric='cosine')
        coherence = 1 - distances.mean() if len(distances) > 0 else 0
        
        # Centroid
        centroid = embs.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        
        # Spread (std of distances from centroid)
        distances_to_centroid = [cosine(emb, centroid) for emb in embs]
        spread = np.std(distances_to_centroid)
        
        # Dimensionality
        try:
            pca = PCA()
            pca.fit(embs)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            dims_90 = np.argmax(cumsum >= 0.9) + 1 if np.any(cumsum >= 0.9) else len(cumsum)
        except:
            dims_90 = -1
        
        print(f"\n{set_name} ({len(codes_with_emb)} procedures):")
        print(f"  Internal Coherence: {coherence:.3f} (1=identical, 0=orthogonal)")
        print(f"  Centroid Norm: {centroid_norm:.3f}")
        print(f"  Spread from Centroid: {spread:.3f}")
        if dims_90 > 0:
            print(f"  Effective Dimensions (90% var): {dims_90}/{embs.shape[1]}")
        
        return centroid
    
    centroid_ab = analyze_set_geometry(ab_only, "AB-only") if len(ab_only) > 0 else None
    centroid_ac = analyze_set_geometry(ac_only, "AC-only") if len(ac_only) > 0 else None
    centroid_abc = analyze_set_geometry(abc_all, "ABC-all") if len(abc_all) > 0 else None
    
    # Compare centroids
    if centroid_ab is not None and centroid_ac is not None:
        sim = 1 - cosine(centroid_ab, centroid_ac)
        print(f"\nCentroid Similarity AB vs AC: {sim:.3f}")

def diagnose_similarity_discrepancy(provider_A_idx, provider_B_idx, provider_C_idx):
    """
    Full diagnostic pipeline
    """
    
    # 1. Get procedure profiles
    proc_df = analyze_provider_procedures(provider_A_idx, provider_B_idx, provider_C_idx)
    
    # 2. Add meaningfulness metrics
    print("\nAnalyzing procedure meaningfulness...")
    proc_df = analyze_procedure_meaningfulness(proc_df)
    
    # 3. Add graph position metrics
    print("Analyzing graph positions...")
    proc_df = analyze_graph_positions(proc_df)
    
    # 4. Compare exclusive overlaps
    ab_only, ac_only, abc_all = compare_exclusive_overlaps(proc_df)
    
    # 5. Analyze embedding contributions
    analyze_embedding_contributions(provider_A_idx, provider_B_idx, provider_C_idx, proc_df)
    
    # 6. Analyze embedding space geometry
    if len(ab_only) > 0 and len(ac_only) > 0:
        analyze_embedding_space_geometry(ab_only, ac_only, abc_all)
    
    # 7. Final diagnosis
    print(f"\n{'='*60}")
    print("FINAL DIAGNOSIS")
    print(f"{'='*60}")
    
    hypotheses_confirmed = []
    
    if len(ab_only) > 0 and len(ac_only) > 0:
        # Check hypotheses
        if ab_only['global_freq_pct'].mean() < ac_only['global_freq_pct'].mean():
            hypotheses_confirmed.append("✓ AB procedures are RARER (more distinctive)")
        else:
            hypotheses_confirmed.append("✗ AB procedures are NOT rarer")
        
        if 'embedding_norm' in ab_only.columns and ab_only['embedding_norm'].mean() > ac_only['embedding_norm'].mean():
            hypotheses_confirmed.append("✓ AB procedures have STRONGER embeddings")
        elif 'embedding_norm' in ab_only.columns:
            hypotheses_confirmed.append("✗ AB procedures do NOT have stronger embeddings")
        
        if 'clustering_coef' in ab_only.columns and ab_only['clustering_coef'].mean() > ac_only['clustering_coef'].mean():
            hypotheses_confirmed.append("✓ AB procedures form TIGHTER CLUSTERS")
        elif 'clustering_coef' in ab_only.columns:
            hypotheses_confirmed.append("✗ AB procedures do NOT form tighter clusters")
        
        if 'co_occurrence_diversity' in ac_only.columns and ac_only['co_occurrence_diversity'].mean() > ab_only['co_occurrence_diversity'].mean():
            hypotheses_confirmed.append("✓ AC procedures are more GENERIC (higher co-occurrence diversity)")
        elif 'co_occurrence_diversity' in ac_only.columns:
            hypotheses_confirmed.append("✗ AC procedures are NOT more generic")
    
    print("\nHypothesis Testing Results:")
    for h in hypotheses_confirmed:
        print(f"  {h}")
    
    # Get similarities
    sim_AB_final = 1 - cosine(final_embeddings[provider_A_idx], final_embeddings[provider_B_idx])
    sim_AC_final = 1 - cosine(final_embeddings[provider_A_idx], final_embeddings[provider_C_idx])
    
    # Get overlaps
    overlap_AB = proc_df['in_AB'].sum()
    overlap_AC = proc_df['in_AC'].sum()
    
    # Conditional summary based on actual pattern
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"Procedure Overlap:")
    print(f"  A-B: {overlap_AB} procedures")
    print(f"  A-C: {overlap_AC} procedures")
    
    print(f"\nFinal Embedding Similarity:")
    print(f"  A-B: {sim_AB_final:.4f}")
    print(f"  A-C: {sim_AC_final:.4f}")
    
    # Determine discrepancy pattern and root cause
    if provider_init_embeddings is not None:
        sim_AB_init = 1 - cosine(provider_init_embeddings[provider_A_idx], 
                                 provider_init_embeddings[provider_B_idx])
        sim_AC_init = 1 - cosine(provider_init_embeddings[provider_A_idx], 
                                 provider_init_embeddings[provider_C_idx])
        
        print(f"\nInitial Embedding Similarity:")
        print(f"  A-B: {sim_AB_init:.4f}")
        print(f"  A-C: {sim_AC_init:.4f}")
    
    print(f"\n{'='*60}")
    print("DISCREPANCY ANALYSIS")
    print(f"{'='*60}")
    
    # Check for discrepancy
    if overlap_AB > overlap_AC and sim_AC_final > sim_AB_final:
        print("✓ DISCREPANCY FOUND: Provider C is more similar to A than B,")
        print(f"  despite B having {overlap_AB - overlap_AC} more shared procedures.")
        
        if provider_init_embeddings is not None:
            if sim_AC_init > sim_AB_init:
                print("\n→ Root Cause: WORD2VEC/NODE2VEC STAGE")
                print("  The discrepancy is already present in initial embeddings.")
                print("  This stems from procedure co-occurrence patterns in the graph.")
            else:
                print("\n→ Root Cause: GAT/FINE-TUNING STAGE")
                print("  Initial embeddings had AB ≥ AC, but GAT flipped it.")
                print("  The attention mechanism learned to weight procedures differently.")
    
    elif overlap_AC > overlap_AB and sim_AB_final > sim_AC_final:
        print("✓ DISCREPANCY FOUND: Provider B is more similar to A than C,")
        print(f"  despite C having {overlap_AC - overlap_AB} more shared procedures.")
        
        if provider_init_embeddings is not None:
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
        if overlap_AB > overlap_AC:
            print(f"  B has more overlap AND higher similarity with A (as expected).")
        else:
            print(f"  C has more overlap AND higher similarity with A (as expected).")
    
    print(f"\nConclusion:")
    print("The embedding model prioritizes procedure co-occurrence patterns and")
    print("learned attention weights over raw procedure overlap counts.")
    
    return proc_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*80)
print("RUNNING FULL DIAGNOSTIC")
print("="*80)

# Run the diagnostic
proc_df = diagnose_similarity_discrepancy(provider_A_idx, provider_B_idx, provider_C_idx)

# Save results
output_filename = f"diagnosis_PIN_{PIN_A}_{PIN_B}_{PIN_C}.csv"
proc_df.to_csv(output_filename, index=False)
print(f"\nDetailed procedure analysis saved to: {output_filename}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
