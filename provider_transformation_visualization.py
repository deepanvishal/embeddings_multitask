"""
PROVIDER EMBEDDING TRANSFORMATION VISUALIZATION
===============================================

Interactive visualization showing how a provider's raw procedure data
transforms through each stage of the embedding pipeline.

Features:
- Select two providers (A and B)
- Compare top 20 procedures with descriptions
- Visualize embeddings at each transformation stage:
  * Specialty-agnostic (averaged)
  * Multi-view (15 specialty views)
  * Reduced (Phase 3A)
  * Final (Phase 3B)

Author: AI Assistant
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import sys

sys.path.append('.')
from procedure_encoder import ProcedureEmbeddingEncoder

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()
print(f"Total providers: {len(all_pins)}")

proc_matrix = load_npz('procedure_vectors.npz')
print(f"Procedure matrix: {proc_matrix.shape}")

pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

try:
    pin_names_df = pd.read_parquet('all_pin_names.parquet')
    pin_to_name = dict(zip(pin_names_df['PIN'], pin_names_df['PIN_name']))
    print(f"Provider names: {len(pin_to_name)}")
except:
    pin_to_name = {pin: f"Provider {pin}" for pin in all_pins}
    print("Provider names file not found, using PINs")

try:
    with open('pin_to_label.pkl', 'rb') as f:
        pin_to_label = pickle.load(f)
    print(f"Labeled providers: {len(pin_to_label)}")
except:
    pin_to_label = {}
    print("Labels file not found")

code_desc_df = pd.read_parquet('code_desc_df.parquet')
code_desc_df = code_desc_df.sort_values('claims', ascending=False)
code_desc_df = code_desc_df.drop_duplicates(subset='code', keep='first')
code_to_desc = dict(zip(code_desc_df['code'], code_desc_df['code_desc']))
print(f"Code descriptions: {len(code_to_desc)}")

procedure_df = pd.read_parquet('procedure_df.parquet')
print(f"Procedure data: {procedure_df.shape}")

procedure_summary = procedure_df.groupby('PIN').agg({
    'code': lambda x: list(x),
    'claims': lambda x: list(x)
}).reset_index()

pin_to_procedure_data = {}
for _, row in procedure_summary.iterrows():
    pin = row['PIN']
    codes = row['code']
    claims = row['claims']
    pin_to_procedure_data[pin] = pd.DataFrame({'code': codes, 'claims': claims})

print(f"Procedure data for {len(pin_to_procedure_data)} providers")

print("\n" + "="*80)
print("INITIALIZING ENCODER")
print("="*80)

encoder = ProcedureEmbeddingEncoder(model_dir='.')
specialty_names = encoder.get_specialty_names()

print("\n" + "="*80)
print("LOADING PRECOMPUTED EMBEDDINGS")
print("="*80)

multiview_embeddings = np.load('phase2_multiview_embeddings.npy')
reduced_embeddings = np.load('phase3a_reduced_embeddings.npy')
final_embeddings = np.load('final_embeddings_128d.npy')

print(f"Multiview: {multiview_embeddings.shape}")
print(f"Reduced: {reduced_embeddings.shape}")
print(f"Final: {final_embeddings.shape}")

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity"""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b + 1e-8)

def get_top_procedures(pin, top_n=20):
    """Get top procedures with descriptions"""
    if pin not in pin_to_procedure_data:
        return pd.DataFrame(columns=['code', 'description', 'claims', '%'])
    
    df = pin_to_procedure_data[pin].copy()
    df = df.groupby('code', as_index=False)['claims'].sum()
    total_claims = df['claims'].sum()
    df['%'] = (df['claims'] / total_claims * 100).round(2)
    
    df['description'] = df['code'].map(code_to_desc).fillna('Unknown')
    
    result = df.nlargest(top_n, 'claims')[['code', 'description', 'claims', '%']]
    return result

def find_common_procedures(pin_a, pin_b, top_n=20):
    """Find common procedures"""
    proc_a = get_top_procedures(pin_a, 50)
    proc_b = get_top_procedures(pin_b, 50)
    
    if len(proc_a) == 0 or len(proc_b) == 0:
        return pd.DataFrame(columns=['code', 'description', 'claims_A', 'claims_B', '%_A', '%_B'])
    
    common = proc_a.merge(proc_b, on=['code', 'description'], suffixes=('_A', '_B'), how='inner')
    common['total_claims'] = common['claims_A'] + common['claims_B']
    
    result = common.nlargest(top_n, 'total_claims')[['code', 'description', 'claims_A', 'claims_B', '%_A', '%_B']]
    return result

def compare_providers(pin_a, pin_b):
    """Compare two providers through transformation stages"""
    
    if pin_a is None or pin_b is None:
        print("Please select both providers")
        return
    
    if pin_a == pin_b:
        print("Please select two different providers")
        return
    
    name_a = pin_to_name.get(pin_a, f"Provider {pin_a}")
    name_b = pin_to_name.get(pin_b, f"Provider {pin_b}")
    label_a = pin_to_label.get(pin_a, "Unknown")
    label_b = pin_to_label.get(pin_b, "Unknown")
    
    idx_a = pin_to_idx[pin_a]
    idx_b = pin_to_idx[pin_b]
    
    clear_output(wait=True)
    
    print("\n" + "="*80)
    print("PROVIDER COMPARISON")
    print("="*80)
    print(f"\nProvider A: {name_a}")
    print(f"  PIN: {pin_a}")
    print(f"  Specialty: {label_a}")
    print(f"\nProvider B: {name_b}")
    print(f"  PIN: {pin_b}")
    print(f"  Specialty: {label_b}")
    
    print("\n" + "="*80)
    print("STEP 1: RAW PROCEDURE DATA")
    print("="*80)
    
    proc_vec_a = proc_matrix[idx_a].toarray().flatten()
    proc_vec_b = proc_matrix[idx_b].toarray().flatten()
    
    n_codes_a = np.sum(proc_vec_a > 0)
    n_codes_b = np.sum(proc_vec_b > 0)
    total_claims_a = np.sum(proc_vec_a)
    total_claims_b = np.sum(proc_vec_b)
    
    print(f"\nProvider A: {n_codes_a} unique procedures, {total_claims_a:.0f} total claims")
    print(f"Provider B: {n_codes_b} unique procedures, {total_claims_b:.0f} total claims")
    
    print("\n" + "="*80)
    print("STEP 2: TOP 20 PROCEDURES")
    print("="*80)
    
    proc_a = get_top_procedures(pin_a, 20)
    proc_b = get_top_procedures(pin_b, 20)
    common_proc = find_common_procedures(pin_a, pin_b, 20)
    
    print(f"\nTop 20 Procedures - {name_a}:")
    if len(proc_a) > 0:
        display(proc_a)
    else:
        print("No procedure data")
    
    print(f"\nTop 20 Procedures - {name_b}:")
    if len(proc_b) > 0:
        display(proc_b)
    else:
        print("No procedure data")
    
    print(f"\nCommon Procedures:")
    if len(common_proc) > 0:
        print(f"Found {len(common_proc)} common procedures in top 50")
        display(common_proc)
    else:
        print("No common procedures in top 50")
    
    print("\n" + "="*80)
    print("STEP 3: SPECIALTY-AGNOSTIC EMBEDDING")
    print("="*80)
    print("(Average across all 15 specialty views)\n")
    
    agnostic_a = encoder.encode_specialty_agnostic(proc_vec_a).flatten()
    agnostic_b = encoder.encode_specialty_agnostic(proc_vec_b).flatten()
    
    agnostic_sim = cosine_similarity(agnostic_a, agnostic_b)
    
    print(f"Embedding dimension: {len(agnostic_a)}")
    print(f"Cosine similarity: {agnostic_sim:.4f}")
    
    plt.figure(figsize=(14, 4))
    plt.plot(agnostic_a, label=f'Provider A ({name_a})', alpha=0.7, linewidth=1.5)
    plt.plot(agnostic_b, label=f'Provider B ({name_b})', alpha=0.7, linewidth=1.5)
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.title(f'Specialty-Agnostic Embeddings (Cosine Sim: {agnostic_sim:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("STEP 4: MULTI-VIEW EMBEDDINGS")
    print("="*80)
    print("(15 specialty views concatenated)\n")
    
    multiview_a = multiview_embeddings[idx_a]
    multiview_b = multiview_embeddings[idx_b]
    
    multiview_sim = cosine_similarity(multiview_a, multiview_b)
    
    print(f"Embedding dimension: {len(multiview_a)}")
    print(f"Cosine similarity: {multiview_sim:.4f}")
    
    plt.figure(figsize=(16, 5))
    plt.plot(multiview_a, label=f'Provider A ({name_a})', alpha=0.7, linewidth=1)
    plt.plot(multiview_b, label=f'Provider B ({name_b})', alpha=0.7, linewidth=1)
    
    latent_dim = encoder.phase1_latent_dim
    for i in range(encoder.num_specialties):
        mid_point = (i * latent_dim) + (latent_dim // 2)
        plt.axvline(x=i * latent_dim, color='gray', linestyle='--', alpha=0.3)
        plt.text(mid_point, plt.ylim()[1] * 0.95, specialty_names[i], 
                ha='center', va='top', fontsize=8, rotation=45)
    
    plt.xlabel('Dimension (grouped by specialty)')
    plt.ylabel('Value')
    plt.title(f'Multi-View Embeddings - 15 Specialty Views (Cosine Sim: {multiview_sim:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("STEP 5: REDUCED EMBEDDINGS (Phase 3A)")
    print("="*80)
    print("(Dimensionality reduction via autoencoder)\n")
    
    reduced_a = reduced_embeddings[idx_a]
    reduced_b = reduced_embeddings[idx_b]
    
    reduced_sim = cosine_similarity(reduced_a, reduced_b)
    
    print(f"Embedding dimension: {len(reduced_a)}")
    print(f"Cosine similarity: {reduced_sim:.4f}")
    
    plt.figure(figsize=(14, 4))
    plt.plot(reduced_a, label=f'Provider A ({name_a})', alpha=0.7, linewidth=1.5)
    plt.plot(reduced_b, label=f'Provider B ({name_b})', alpha=0.7, linewidth=1.5)
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.title(f'Reduced Embeddings (Cosine Sim: {reduced_sim:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("STEP 6: FINAL EMBEDDINGS (Phase 3B)")
    print("="*80)
    print("(Supervised contrastive compression)\n")
    
    final_a = final_embeddings[idx_a]
    final_b = final_embeddings[idx_b]
    
    final_sim = cosine_similarity(final_a, final_b)
    euclidean_dist = np.linalg.norm(final_a - final_b)
    
    print(f"Embedding dimension: {len(final_a)}")
    print(f"Cosine similarity: {final_sim:.4f}")
    print(f"Euclidean distance: {euclidean_dist:.4f}")
    
    plt.figure(figsize=(14, 4))
    plt.plot(final_a, label=f'Provider A ({name_a})', alpha=0.7, linewidth=1.5)
    plt.plot(final_b, label=f'Provider B ({name_b})', alpha=0.7, linewidth=1.5)
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.title(f'Final Embeddings (Cosine Sim: {final_sim:.4f}, Euclidean: {euclidean_dist:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("TRANSFORMATION SUMMARY")
    print("="*80)
    print(f"\nCosine Similarity at Each Stage:")
    print(f"  Specialty-Agnostic: {agnostic_sim:.4f}")
    print(f"  Multi-View:         {multiview_sim:.4f}")
    print(f"  Reduced:            {reduced_sim:.4f}")
    print(f"  Final:              {final_sim:.4f}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

print("\n" + "="*80)
print("CREATING INTERACTIVE WIDGETS")
print("="*80)

dropdown_options = [('-- Select Provider --', None)] + \
                   [(f"{pin_to_name.get(pin, 'Unknown')} ({pin})", pin) 
                    for pin in all_pins]
dropdown_options = sorted(dropdown_options[1:], key=lambda x: x[0])
dropdown_options = [('-- Select Provider --', None)] + dropdown_options

provider_a_dropdown = widgets.Dropdown(
    options=dropdown_options,
    description='Provider A:',
    disabled=False,
    layout=widgets.Layout(width='80%'),
    style={'description_width': '150px'}
)

provider_b_dropdown = widgets.Dropdown(
    options=dropdown_options,
    description='Provider B:',
    disabled=False,
    layout=widgets.Layout(width='80%'),
    style={'description_width': '150px'}
)

compare_button = widgets.Button(
    description='Compare Providers',
    disabled=False,
    button_style='primary',
    layout=widgets.Layout(width='200px')
)

output_area = widgets.Output()

def on_compare_clicked(b):
    """Handle compare button click"""
    with output_area:
        pin_a = provider_a_dropdown.value
        pin_b = provider_b_dropdown.value
        compare_providers(pin_a, pin_b)

compare_button.on_click(on_compare_clicked)

print("\n" + "="*80)
print("PROVIDER EMBEDDING TRANSFORMATION VISUALIZATION")
print("="*80)
print(f"\nAvailable providers: {len(all_pins):,}")
print(f"Labeled providers: {len(pin_to_label):,}")
print("\nSelect two providers and click 'Compare Providers'")
print("\nVisualization stages:")
print("  1. Raw procedure data comparison")
print("  2. Top 20 procedures with descriptions")
print("  3. Specialty-agnostic embedding (averaged)")
print("  4. Multi-view embeddings (15 specialty views)")
print("  5. Reduced embeddings (Phase 3A)")
print("  6. Final embeddings (Phase 3B)")
print("\nSystem ready!")

display(widgets.VBox([
    provider_a_dropdown,
    provider_b_dropdown,
    compare_button,
    output_area
]))
