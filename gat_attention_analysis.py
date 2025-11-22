"""
GAT Attention Weight Analysis
Extracts and analyzes what the GAT is actually learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import seaborn as sns
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
print("GAT ATTENTION WEIGHT EXTRACTION AND ANALYSIS")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# REBUILD THE GAT MODEL (same architecture as in training)
# ============================================================================

class ProviderGAT(nn.Module):
    def __init__(self, code_embedding_dim, provider_embedding_dim, num_specialties, num_heads=4, dropout=0.3):
        super(ProviderGAT, self).__init__()
        self.num_heads = num_heads
        self.provider_embedding_dim = provider_embedding_dim
        self.head_dim = provider_embedding_dim
        
        # Multi-head attention
        self.W_heads = nn.ModuleList([
            nn.Linear(code_embedding_dim, self.head_dim, bias=False) 
            for _ in range(num_heads)
        ])
        self.a_heads = nn.ParameterList([
            nn.Parameter(torch.randn(2 * self.head_dim, 1))
            for _ in range(num_heads)
        ])
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.specialty_classifier = nn.Linear(provider_embedding_dim * num_heads, num_specialties)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, provider_emb, code_embs, return_attention=False):
        batch_size = 1
        n_codes = code_embs.shape[0]
        
        head_outputs = []
        attention_weights_all = []
        
        for head_idx in range(self.num_heads):
            W = self.W_heads[head_idx]
            a = self.a_heads[head_idx]
            
            # Transform embeddings
            provider_h = W(provider_emb)
            code_h = W(code_embs)
            
            # Compute attention scores
            provider_repeated = provider_h.repeat(n_codes, 1)
            concat = torch.cat([provider_repeated, code_h], dim=1)
            e = self.leaky_relu(concat @ a)
            
            # Apply softmax to get attention weights
            alpha = F.softmax(e, dim=0)
            attention_weights_all.append(alpha.detach().cpu().numpy())
            
            # Aggregate with attention weights
            attended = (alpha * code_h).sum(dim=0, keepdim=True)
            head_outputs.append(attended)
        
        # Concatenate all heads
        multi_head_output = torch.cat(head_outputs, dim=1)
        
        # Classify specialty
        multi_head_output = self.dropout(multi_head_output)
        specialty_logits = self.specialty_classifier(multi_head_output)
        
        if return_attention:
            return specialty_logits.squeeze(0), multi_head_output, np.array(attention_weights_all)
        else:
            return specialty_logits.squeeze(0), multi_head_output

# ============================================================================
# LOAD ALL NECESSARY DATA
# ============================================================================

print("\nLoading data...")

# Load basic data
proc_matrix = load_npz('procedure_vectors.npz')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()
final_embeddings = np.load('me2vec_provider_embeddings.npy')

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

with open('specialty_code_mappings.pkl', 'rb') as f:
    specialty_mappings = pickle.load(f)

# Load embeddings
provider_init_embeddings = np.load('provider_init_embeddings.npy')
code_embeddings = np.load('code_embeddings_dict.npy', allow_pickle=True).item()

# Load metadata to get model configuration
with open('me2vec_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

CODE_EMBEDDING_DIM = metadata['code_embedding_dim']
PROVIDER_EMBEDDING_DIM = metadata['provider_embedding_dim']
NUM_HEADS = metadata['num_heads']
num_specialties = metadata['num_specialties']

print(f"Model configuration:")
print(f"  Code embedding dim: {CODE_EMBEDDING_DIM}")
print(f"  Provider embedding dim: {PROVIDER_EMBEDDING_DIM}")
print(f"  Number of heads: {NUM_HEADS}")
print(f"  Number of specialties: {num_specialties}")

# Get specialty codes
all_specialty_codes = sorted(list(set().union(*specialty_mappings['code_indices'].values())))
proc_matrix_filtered = proc_matrix[:, all_specialty_codes].tocsr()

# Create mappings
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

# Validate PINs
if PIN_A not in pin_to_idx or PIN_B not in pin_to_idx or PIN_C not in pin_to_idx:
    print("ERROR: One or more PINs not found!")
    exit(1)

provider_A_idx = pin_to_idx[PIN_A]
provider_B_idx = pin_to_idx[PIN_B]
provider_C_idx = pin_to_idx[PIN_C]

print(f"\nProviders:")
print(f"  A: PIN {PIN_A} (index {provider_A_idx})")
print(f"  B: PIN {PIN_B} (index {provider_B_idx})")
print(f"  C: PIN {PIN_C} (index {provider_C_idx})")

# ============================================================================
# LOAD THE TRAINED GAT MODEL
# ============================================================================

print("\n" + "="*60)
print("LOADING TRAINED GAT MODEL")
print("="*60)

# Initialize model
model = ProviderGAT(CODE_EMBEDDING_DIM, PROVIDER_EMBEDDING_DIM, 
                    num_specialties, NUM_HEADS, dropout=0.3).to(device)

# Try to load saved model weights
try:
    model.load_state_dict(torch.load('gat_model.pth', map_location=device))
    print("✓ Loaded saved GAT model weights")
except:
    print("✗ No saved model found - using random weights")
    print("  To get real attention weights, save the model in training:")
    print("  torch.save(model.state_dict(), 'gat_model.pth')")

model.eval()

# ============================================================================
# EXTRACT PROCEDURES FOR EACH PROVIDER
# ============================================================================

def get_provider_procedures(provider_idx):
    """Extract procedures and their counts for a provider"""
    row = proc_matrix_filtered[provider_idx]
    procedures = {}
    code_list = []
    for idx, count in zip(row.indices, row.data):
        code = all_specialty_codes[idx]
        procedures[code] = count
        code_list.append(code)
    return procedures, code_list

A_procedures, A_code_list = get_provider_procedures(provider_A_idx)
B_procedures, B_code_list = get_provider_procedures(provider_B_idx)
C_procedures, C_code_list = get_provider_procedures(provider_C_idx)

print(f"Provider A: {len(A_procedures)} procedures")
print(f"Provider B: {len(B_procedures)} procedures")
print(f"Provider C: {len(C_procedures)} procedures")

# ============================================================================
# EXTRACT ATTENTION WEIGHTS FOR EACH PROVIDER
# ============================================================================

print("\n" + "="*60)
print("EXTRACTING ATTENTION WEIGHTS")
print("="*60)

def get_attention_weights(provider_idx, procedure_codes):
    """Get GAT attention weights for a provider's procedures"""
    
    # Get initial embedding
    provider_emb = torch.FloatTensor(provider_init_embeddings[provider_idx]).unsqueeze(0).to(device)
    
    # Get code embeddings
    code_emb_list = []
    valid_codes = []
    for code in procedure_codes:
        if code in code_embeddings:
            code_emb_list.append(code_embeddings[code])
            valid_codes.append(code)
    
    if len(code_emb_list) == 0:
        return None, None, None
    
    code_embs = torch.FloatTensor(np.array(code_emb_list)).to(device)
    
    # Get attention weights
    with torch.no_grad():
        _, _, attention_weights = model(provider_emb, code_embs, return_attention=True)
    
    return attention_weights, valid_codes, code_embs

# Get attention for each provider
print("\nProvider A attention...")
A_attention, A_valid_codes, A_code_embs = get_attention_weights(provider_A_idx, A_code_list)

print("Provider B attention...")
B_attention, B_valid_codes, B_code_embs = get_attention_weights(provider_B_idx, B_code_list)

print("Provider C attention...")
C_attention, C_valid_codes, C_code_embs = get_attention_weights(provider_C_idx, C_code_list)

# ============================================================================
# ANALYZE ATTENTION PATTERNS
# ============================================================================

print("\n" + "="*60)
print("ATTENTION WEIGHT ANALYSIS")
print("="*60)

# For each provider, find top procedures by attention
def analyze_provider_attention(attention_weights, valid_codes, provider_name):
    """Analyze which procedures get highest attention"""
    
    print(f"\n{provider_name} Top Procedures by Attention:")
    
    # Average attention across heads
    avg_attention = attention_weights.mean(axis=0).squeeze()
    
    # Get top 10 procedures by attention
    top_indices = np.argsort(avg_attention)[-10:][::-1]
    
    attention_data = []
    for idx in top_indices:
        code = valid_codes[idx]
        att_weight = avg_attention[idx]
        
        # Get procedure info
        code_idx = all_specialty_codes.index(code)
        providers_with_code = (proc_matrix_filtered[:, code_idx] > 0).sum()
        global_freq = providers_with_code / proc_matrix_filtered.shape[0]
        
        attention_data.append({
            'code': code,
            'attention': att_weight,
            'global_freq': global_freq,
            'embedding_norm': np.linalg.norm(code_embeddings[code]) if code in code_embeddings else 0
        })
    
    df = pd.DataFrame(attention_data)
    print(df.to_string(index=False))
    
    # Attention concentration
    top_5_attention = np.sort(avg_attention)[-5:].sum()
    print(f"\nAttention concentration:")
    print(f"  Top 5 procedures capture {top_5_attention:.1%} of total attention")
    print(f"  Gini coefficient: {calculate_gini(avg_attention):.3f} (0=uniform, 1=concentrated)")
    
    return avg_attention, df

def calculate_gini(x):
    """Calculate Gini coefficient for concentration"""
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    return (2 * np.sum((np.arange(n) + 1) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n

# Analyze each provider
if A_attention is not None:
    A_avg_attention, A_top_df = analyze_provider_attention(A_attention, A_valid_codes, "Provider A")

if B_attention is not None:
    B_avg_attention, B_top_df = analyze_provider_attention(B_attention, B_valid_codes, "Provider B")

if C_attention is not None:
    C_avg_attention, C_top_df = analyze_provider_attention(C_attention, C_valid_codes, "Provider C")

# ============================================================================
# COMPARE ATTENTION ON SHARED VS EXCLUSIVE PROCEDURES
# ============================================================================

print("\n" + "="*60)
print("ATTENTION ON SHARED VS EXCLUSIVE PROCEDURES")
print("="*60)

# Get procedure sets
A_set = set(A_valid_codes)
B_set = set(B_valid_codes)
C_set = set(C_valid_codes)

AB_shared = A_set & B_set
AC_shared = A_set & C_set
AB_only = AB_shared - C_set
AC_only = AC_shared - B_set
ABC_all = A_set & B_set & C_set

# Analyze attention on different sets for Provider A
if A_attention is not None:
    print("\nProvider A attention distribution:")
    
    # Get attention for different procedure groups
    ab_only_attention = []
    ac_only_attention = []
    abc_all_attention = []
    
    for i, code in enumerate(A_valid_codes):
        att = A_avg_attention[i]
        if code in AB_only:
            ab_only_attention.append(att)
        elif code in AC_only:
            ac_only_attention.append(att)
        elif code in ABC_all:
            abc_all_attention.append(att)
    
    if ab_only_attention:
        print(f"\nAB-only procedures ({len(ab_only_attention)} codes):")
        print(f"  Mean attention: {np.mean(ab_only_attention):.4f}")
        print(f"  Total attention: {np.sum(ab_only_attention):.4f}")
        
    if ac_only_attention:
        print(f"\nAC-only procedures ({len(ac_only_attention)} codes):")
        print(f"  Mean attention: {np.mean(ac_only_attention):.4f}")
        print(f"  Total attention: {np.sum(ac_only_attention):.4f}")
    
    if abc_all_attention:
        print(f"\nABC-all procedures ({len(abc_all_attention)} codes):")
        print(f"  Mean attention: {np.mean(abc_all_attention):.4f}")
        print(f"  Total attention: {np.sum(abc_all_attention):.4f}")
    
    # Statistical comparison
    if ab_only_attention and ac_only_attention:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(ab_only_attention, ac_only_attention)
        print(f"\nAB-only vs AC-only attention (t-test):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        
        if np.mean(ab_only_attention) > np.mean(ac_only_attention):
            print("  → GAT pays MORE attention to AB-only procedures")
        else:
            print("  → GAT pays MORE attention to AC-only procedures")

# ============================================================================
# HEAD-BY-HEAD ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("HEAD-BY-HEAD ATTENTION ANALYSIS")
print("="*60)

if A_attention is not None:
    print("\nProvider A - Attention variance across heads:")
    
    for head_idx in range(NUM_HEADS):
        head_attention = A_attention[head_idx].squeeze()
        top_5_indices = np.argsort(head_attention)[-5:][::-1]
        
        print(f"\nHead {head_idx + 1} top 5 procedures:")
        for idx in top_5_indices:
            code = A_valid_codes[idx]
            att = head_attention[idx]
            print(f"  {code}: {att:.4f}")
        
        # Calculate head specialization
        gini = calculate_gini(head_attention)
        print(f"  Concentration (Gini): {gini:.3f}")

# ============================================================================
# SIMULATE WHAT HAPPENS WITHOUT GAT
# ============================================================================

print("\n" + "="*60)
print("SIMULATING EMBEDDINGS WITHOUT GAT")
print("="*60)

# Calculate what embeddings would be with uniform attention
def calculate_uniform_attention_embedding(provider_idx, procedure_codes):
    """Calculate embedding if all procedures had equal weight"""
    
    embedding = np.zeros(CODE_EMBEDDING_DIM * NUM_HEADS)
    total_weight = 0
    
    for code in procedure_codes:
        if code in code_embeddings:
            # Simulate uniform attention - just average
            for head in range(NUM_HEADS):
                start_idx = head * CODE_EMBEDDING_DIM
                end_idx = (head + 1) * CODE_EMBEDDING_DIM
                embedding[start_idx:end_idx] += code_embeddings[code]
            total_weight += 1
    
    if total_weight > 0:
        embedding /= total_weight
    
    return embedding

# Calculate uniform embeddings
A_uniform = calculate_uniform_attention_embedding(provider_A_idx, A_code_list)
B_uniform = calculate_uniform_attention_embedding(provider_B_idx, B_code_list)
C_uniform = calculate_uniform_attention_embedding(provider_C_idx, C_code_list)

# Compare similarities
from scipy.spatial.distance import cosine

print("Similarities with UNIFORM attention (no GAT bias):")
sim_AB_uniform = 1 - cosine(A_uniform, B_uniform)
sim_AC_uniform = 1 - cosine(A_uniform, C_uniform)
print(f"  A-B: {sim_AB_uniform:.4f}")
print(f"  A-C: {sim_AC_uniform:.4f}")

print("\nActual similarities (with GAT):")
sim_AB_actual = 1 - cosine(final_embeddings[provider_A_idx], final_embeddings[provider_B_idx])
sim_AC_actual = 1 - cosine(final_embeddings[provider_A_idx], final_embeddings[provider_C_idx])
print(f"  A-B: {sim_AB_actual:.4f}")
print(f"  A-C: {sim_AC_actual:.4f}")

print("\nGAT impact:")
print(f"  A-B similarity change: {sim_AB_actual - sim_AB_uniform:+.4f}")
print(f"  A-C similarity change: {sim_AC_actual - sim_AC_uniform:+.4f}")

# ============================================================================
# IDENTIFY PROBLEMATIC PROCEDURES
# ============================================================================

print("\n" + "="*60)
print("IDENTIFYING PROBLEMATIC PROCEDURES")
print("="*60)

if A_attention is not None and B_attention is not None and C_attention is not None:
    # Find procedures that get very different attention between B and C
    
    problematic_procedures = []
    
    # Check AB-only procedures
    for code in AB_only:
        if code in A_valid_codes and code in B_valid_codes:
            a_idx = A_valid_codes.index(code)
            b_idx = B_valid_codes.index(code)
            
            a_att = A_avg_attention[a_idx]
            b_att = B_avg_attention[b_idx] if B_attention is not None else 0
            
            avg_attention = (a_att + b_att) / 2
            
            problematic_procedures.append({
                'code': code,
                'set': 'AB-only',
                'A_attention': a_att,
                'B_attention': b_att,
                'avg_attention': avg_attention
            })
    
    # Check AC-only procedures  
    for code in AC_only:
        if code in A_valid_codes and code in C_valid_codes:
            a_idx = A_valid_codes.index(code)
            c_idx = C_valid_codes.index(code)
            
            a_att = A_avg_attention[a_idx]
            c_att = C_avg_attention[c_idx] if C_attention is not None else 0
            
            avg_attention = (a_att + c_att) / 2
            
            problematic_procedures.append({
                'code': code,
                'set': 'AC-only',
                'A_attention': a_att,
                'C_attention': c_att,
                'avg_attention': avg_attention
            })
    
    if problematic_procedures:
        prob_df = pd.DataFrame(problematic_procedures)
        
        # Sort by average attention to find most influential
        prob_df = prob_df.sort_values('avg_attention', ascending=False)
        
        print("\nMost influential exclusive procedures:")
        print("\nTop AB-only procedures by attention:")
        ab_subset = prob_df[prob_df['set'] == 'AB-only'].head(5)
        if len(ab_subset) > 0:
            print(ab_subset[['code', 'A_attention', 'B_attention', 'avg_attention']].to_string(index=False))
            print(f"Total attention on top 5 AB-only: {ab_subset['avg_attention'].sum():.4f}")
        
        print("\nTop AC-only procedures by attention:")
        ac_subset = prob_df[prob_df['set'] == 'AC-only'].head(5)
        if len(ac_subset) > 0:
            print(ac_subset[['code', 'A_attention', 'C_attention', 'avg_attention']].to_string(index=False))
            print(f"Total attention on top 5 AC-only: {ac_subset['avg_attention'].sum():.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*60)
print("CREATING ATTENTION VISUALIZATIONS")
print("="*60)

if A_attention is not None:
    # Create attention heatmap for Provider A
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Attention distribution
    axes[0].plot(sorted(A_avg_attention, reverse=True), 'b-', linewidth=2)
    axes[0].set_xlabel('Procedure Rank')
    axes[0].set_ylabel('Attention Weight')
    axes[0].set_title('Provider A: Attention Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Head diversity
    head_correlations = np.zeros((NUM_HEADS, NUM_HEADS))
    for i in range(NUM_HEADS):
        for j in range(NUM_HEADS):
            corr = np.corrcoef(A_attention[i].squeeze(), A_attention[j].squeeze())[0, 1]
            head_correlations[i, j] = corr
    
    im = axes[1].imshow(head_correlations, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Head Attention Correlation')
    axes[1].set_xlabel('Head')
    axes[1].set_ylabel('Head')
    plt.colorbar(im, ax=axes[1])
    
    # Plot 3: AB vs AC attention comparison
    if ab_only_attention and ac_only_attention:
        axes[2].violinplot([ab_only_attention, ac_only_attention], positions=[1, 2])
        axes[2].set_xticks([1, 2])
        axes[2].set_xticklabels(['AB-only', 'AC-only'])
        axes[2].set_ylabel('Attention Weight')
        axes[2].set_title('Attention on Exclusive Procedures')
        axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('GAT Attention Analysis for Provider A', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'gat_attention_analysis_{PIN_A}.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: gat_attention_analysis_{PIN_A}.png")
    plt.close()

# ============================================================================
# SAVE DETAILED RESULTS
# ============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save attention weights for all providers
if A_attention is not None:
    attention_data = {
        'provider_A_codes': A_valid_codes,
        'provider_A_attention': A_avg_attention.tolist(),
        'provider_A_attention_per_head': A_attention.tolist()
    }
    
    if B_attention is not None:
        attention_data['provider_B_codes'] = B_valid_codes
        attention_data['provider_B_attention'] = B_avg_attention.tolist()
        
    if C_attention is not None:
        attention_data['provider_C_codes'] = C_valid_codes
        attention_data['provider_C_attention'] = C_avg_attention.tolist()
    
    import json
    with open(f'gat_attention_weights_{PIN_A}_{PIN_B}_{PIN_C}.json', 'w') as f:
        json.dump(attention_data, f, indent=2)
    print(f"Saved attention weights to: gat_attention_weights_{PIN_A}_{PIN_B}_{PIN_C}.json")

# Save summary
summary = {
    'PIN_A': PIN_A,
    'PIN_B': PIN_B,
    'PIN_C': PIN_C,
    'sim_AB_actual': sim_AB_actual,
    'sim_AC_actual': sim_AC_actual,
    'sim_AB_uniform': sim_AB_uniform,
    'sim_AC_uniform': sim_AC_uniform,
    'GAT_impact_AB': sim_AB_actual - sim_AB_uniform,
    'GAT_impact_AC': sim_AC_actual - sim_AC_uniform
}

if ab_only_attention and ac_only_attention:
    summary['mean_attention_AB_only'] = np.mean(ab_only_attention)
    summary['mean_attention_AC_only'] = np.mean(ac_only_attention)
    summary['total_attention_AB_only'] = np.sum(ab_only_attention)
    summary['total_attention_AC_only'] = np.sum(ac_only_attention)

pd.DataFrame([summary]).to_csv(f'gat_analysis_summary_{PIN_A}_{PIN_B}_{PIN_C}.csv', index=False)
print(f"Saved summary to: gat_analysis_summary_{PIN_A}_{PIN_B}_{PIN_C}.csv")

print("\n" + "="*80)
print("GAT ANALYSIS COMPLETE")
print("="*80)

print("\nKEY FINDINGS:")
if 'mean_attention_AB_only' in summary and 'mean_attention_AC_only' in summary:
    if summary['mean_attention_AB_only'] > summary['mean_attention_AC_only']:
        print("  ✗ GAT gives MORE attention to AB-only procedures")
        print("    This SHOULD increase A-B similarity but doesn't explain the discrepancy")
    else:
        print("  ✓ GAT gives MORE attention to AC-only procedures")  
        print("    This explains why A-C similarity is higher despite fewer shared procedures")

print(f"\n  Without GAT bias (uniform attention):")
print(f"    A-B would be: {sim_AB_uniform:.4f}")
print(f"    A-C would be: {sim_AC_uniform:.4f}")
print(f"  GAT changes this to:")
print(f"    A-B becomes: {sim_AB_actual:.4f} ({sim_AB_actual - sim_AB_uniform:+.4f})")
print(f"    A-C becomes: {sim_AC_actual:.4f} ({sim_AC_actual - sim_AC_uniform:+.4f})")
