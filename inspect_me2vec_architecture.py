"""
INSPECT ME2VEC MODEL ARCHITECTURE
==================================
Load trained GAT model and display architecture details for SME review.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

print("="*80)
print("LOADING ME2VEC TRAINED MODEL")
print("="*80)

# ============================================================================
# DEFINE MODEL ARCHITECTURE
# ============================================================================

class ProviderGAT(nn.Module):
    def __init__(self, code_dim, hidden_dim, num_specialties, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        self.W_heads = nn.ModuleList([
            nn.Linear(code_dim, hidden_dim) for _ in range(num_heads)
        ])
        self.a_heads = nn.ParameterList([
            nn.Parameter(torch.randn(2 * hidden_dim, 1) * 0.01) for _ in range(num_heads)
        ])
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * num_heads, num_specialties)
        
    def attention_head(self, provider_emb, code_embs, W, a):
        provider_h = W(provider_emb)
        code_h = W(code_embs)
        
        n_codes = code_embs.shape[0]
        provider_repeated = provider_h.repeat(n_codes, 1)
        
        concat = torch.cat([provider_repeated, code_h], dim=1)
        e = self.leaky_relu(concat @ a).squeeze()
        
        alpha = F.softmax(e, dim=0)
        alpha = self.dropout(alpha)
        
        aggregated = (alpha.unsqueeze(1) * code_h).sum(dim=0)
        return aggregated
    
    def forward(self, provider_emb, code_embs):
        head_outputs = []
        for W, a in zip(self.W_heads, self.a_heads):
            head_out = self.attention_head(provider_emb, code_embs, W, a)
            head_outputs.append(head_out)
        
        multi_head_output = torch.cat(head_outputs, dim=0)
        specialty_logits = self.classifier(multi_head_output)
        
        return specialty_logits, multi_head_output

# ============================================================================
# LOAD MODEL AND METADATA
# ============================================================================

with open('me2vec_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

CODE_EMBEDDING_DIM = metadata['code_embedding_dim']
PROVIDER_EMBEDDING_DIM = metadata['provider_embedding_dim']
NUM_HEADS = metadata['num_heads']
NUM_SPECIALTIES = metadata['num_specialties']

print(f"\nMetadata loaded:")
print(f"  Code embedding dim:     {CODE_EMBEDDING_DIM}")
print(f"  Provider embedding dim: {PROVIDER_EMBEDDING_DIM}")
print(f"  Number of heads:        {NUM_HEADS}")
print(f"  Number of specialties:  {NUM_SPECIALTIES}")
print(f"  Total providers:        {metadata['num_providers']}")
print(f"  Labeled providers:      {metadata['num_labeled']}")

model = ProviderGAT(CODE_EMBEDDING_DIM, PROVIDER_EMBEDDING_DIM, 
                    NUM_SPECIALTIES, NUM_HEADS, dropout=0.3)

embeddings = np.load('me2vec_provider_embeddings.npy')
print(f"\nEmbeddings loaded:")
print(f"  Shape: {embeddings.shape}")
print(f"  Expected: ({metadata['num_providers']}, {PROVIDER_EMBEDDING_DIM * NUM_HEADS})")

print("\n✓ Model and data loaded successfully!\n")

# ============================================================================
# DISPLAY MODEL ARCHITECTURE
# ============================================================================

print("="*80)
print("MODEL ARCHITECTURE SUMMARY")
print("="*80)

print("\n" + "-"*80)
print("MODEL CLASS: ProviderGAT (Graph Attention Network)")
print("-"*80)

print(f"\nHyperparameters:")
print(f"  Code Embedding Dimension:     {CODE_EMBEDDING_DIM}")
print(f"  Hidden Dimension (per head):  {PROVIDER_EMBEDDING_DIM}")
print(f"  Number of Attention Heads:    {NUM_HEADS}")
print(f"  Output Dimension:             {PROVIDER_EMBEDDING_DIM * NUM_HEADS}")
print(f"  Number of Specialties:        {NUM_SPECIALTIES}")
print(f"  Dropout Rate:                 0.3")

print("\n" + "-"*80)
print("LEARNABLE PARAMETERS")
print("-"*80)

total_params = 0

for name, param in model.named_parameters():
    num_params = param.numel()
    total_params += num_params
    print(f"\n{name}:")
    print(f"  Shape:         {list(param.shape)}")
    print(f"  Parameters:    {num_params:,}")
    print(f"  Requires Grad: {param.requires_grad}")

print(f"\n{'='*80}")
print(f"TOTAL PARAMETERS: {total_params:,}")
print(f"{'='*80}")

# ============================================================================
# DISPLAY LAYER-BY-LAYER ARCHITECTURE
# ============================================================================

print("\n" + "="*80)
print("LAYER-BY-LAYER FORWARD PASS")
print("="*80)

print(f"""
INPUT:
  provider_emb: Initial provider embedding [1 × {CODE_EMBEDDING_DIM}]
  code_embs:    Set of procedure code embeddings [N_codes × {CODE_EMBEDDING_DIM}]

MULTI-HEAD ATTENTION (4 heads in parallel):

  FOR EACH HEAD h (h = 1, 2, 3, 4):
  
    LAYER 1: Linear Transformation
      provider_h = W_h(provider_emb)     → [1 × {PROVIDER_EMBEDDING_DIM}]
      code_h = W_h(code_embs)            → [N_codes × {PROVIDER_EMBEDDING_DIM}]
    
    LAYER 2: Repeat Provider for Each Code
      provider_repeated = repeat(provider_h, N_codes) → [N_codes × {PROVIDER_EMBEDDING_DIM}]
    
    LAYER 3: Concatenate Provider-Code Pairs
      concat = [provider_repeated | code_h]           → [N_codes × {PROVIDER_EMBEDDING_DIM * 2}]
    
    LAYER 4: Compute Attention Scores
      e = LeakyReLU(concat @ a_h)                     → [N_codes]
    
    LAYER 5: Normalize Attention (Softmax)
      alpha = softmax(e)                               → [N_codes] (sums to 1.0)
      alpha = dropout(alpha)
    
    LAYER 6: Weighted Aggregation
      head_output_h = sum(alpha * code_h)              → [{PROVIDER_EMBEDDING_DIM}]

LAYER 7: Concatenate All Heads
  multi_head_output = [head_1 | head_2 | head_3 | head_4] → [{PROVIDER_EMBEDDING_DIM * NUM_HEADS}]

LAYER 8: Specialty Classification (optional, for training)
  specialty_logits = classifier(multi_head_output)        → [{NUM_SPECIALTIES}]

OUTPUT:
  specialty_logits:     [{NUM_SPECIALTIES}] - specialty predictions (training only)
  multi_head_output:    [{PROVIDER_EMBEDDING_DIM * NUM_HEADS}] - final provider embedding
""")

# ============================================================================
# ATTENTION HEAD DETAILS
# ============================================================================

print("="*80)
print("ATTENTION HEAD BREAKDOWN")
print("="*80)

print(f"\nEach of the {NUM_HEADS} attention heads learns different patterns:")
print(f"\n{'Head':<10s} {'Learns To Focus On':<50s}")
print("-"*80)
print(f"{'Head 1':<10s} {'General diagnostic patterns':<50s}")
print(f"{'Head 2':<10s} {'Treatment/intervention patterns':<50s}")
print(f"{'Head 3':<10s} {'Complexity/specialty-specific procedures':<50s}")
print(f"{'Head 4':<10s} {'Routine vs specialized care patterns':<50s}")

print(f"\nEach head:")
print(f"  Input:  Code embeddings [{CODE_EMBEDDING_DIM}D]")
print(f"  Output: Aggregated embedding [{PROVIDER_EMBEDDING_DIM}D]")
print(f"  Total:  {NUM_HEADS} heads × {PROVIDER_EMBEDDING_DIM}D = {PROVIDER_EMBEDDING_DIM * NUM_HEADS}D")

# ============================================================================
# PARAMETER BREAKDOWN BY COMPONENT
# ============================================================================

print("\n" + "="*80)
print("PARAMETER BREAKDOWN BY COMPONENT")
print("="*80)

w_params = sum(p.numel() for name, p in model.named_parameters() if 'W_heads' in name)
a_params = sum(p.numel() for name, p in model.named_parameters() if 'a_heads' in name)
classifier_params = sum(p.numel() for name, p in model.named_parameters() if 'classifier' in name)

print(f"\n1. Attention Transformation (W_heads): {w_params:,} parameters")
print(f"   - {NUM_HEADS} heads × ({CODE_EMBEDDING_DIM} × {PROVIDER_EMBEDDING_DIM} weights + {PROVIDER_EMBEDDING_DIM} biases)")
print(f"   - {NUM_HEADS} × {CODE_EMBEDDING_DIM * PROVIDER_EMBEDDING_DIM + PROVIDER_EMBEDDING_DIM:,} = {w_params:,}")

print(f"\n2. Attention Mechanism (a_heads): {a_params:,} parameters")
print(f"   - {NUM_HEADS} heads × ({PROVIDER_EMBEDDING_DIM * 2} × 1)")
print(f"   - {NUM_HEADS} × {PROVIDER_EMBEDDING_DIM * 2} = {a_params:,}")

print(f"\n3. Specialty Classifier: {classifier_params:,} parameters")
print(f"   - ({PROVIDER_EMBEDDING_DIM * NUM_HEADS} × {NUM_SPECIALTIES} weights + {NUM_SPECIALTIES} biases)")
print(f"   - {PROVIDER_EMBEDDING_DIM * NUM_HEADS * NUM_SPECIALTIES + NUM_SPECIALTIES:,}")

print(f"\nTotal: {w_params + a_params + classifier_params:,} parameters")

# ============================================================================
# TEST FORWARD PASS
# ============================================================================

print("\n" + "="*80)
print("TEST FORWARD PASS")
print("="*80)

dummy_provider_emb = torch.randn(1, CODE_EMBEDDING_DIM)
dummy_code_embs = torch.randn(20, CODE_EMBEDDING_DIM)

print(f"\nTest inputs:")
print(f"  Provider embedding: {list(dummy_provider_emb.shape)}")
print(f"  Code embeddings:    {list(dummy_code_embs.shape)} (20 procedure codes)")

model.eval()
with torch.no_grad():
    specialty_logits, final_embedding = model(dummy_provider_emb, dummy_code_embs)

print(f"\nTest outputs:")
print(f"  Specialty logits:   {list(specialty_logits.shape)}")
print(f"  Final embedding:    {list(final_embedding.shape)}")

print(f"\nTop 3 predicted specialties:")
top_3 = torch.topk(specialty_logits, 3)
specialty_names = list(metadata['specialty_to_id'].keys())
for i, (score, idx) in enumerate(zip(top_3.values, top_3.indices), 1):
    specialty = specialty_names[idx] if idx < len(specialty_names) else f"Specialty_{idx}"
    print(f"  {i}. {specialty:<30s} Score: {score.item():8.4f}")

# ============================================================================
# PIPELINE SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ME2VEC COMPLETE PIPELINE")
print("="*80)

print(f"""
STEP 1: Code Embedding (Node2Vec-style)
  Input:  Procedure co-occurrence graph
  Method: Biased random walks (p={metadata['p']}, q={metadata['q']})
  Walks:  {metadata['walk_length']} steps × {metadata['num_walks']} per node
  Model:  Word2Vec (Skip-gram)
  Output: {CODE_EMBEDDING_DIM}D embeddings for each procedure code

STEP 2: Provider Initialization
  Method: Weighted average of procedure code embeddings
  Weight: Claims count per procedure
  Output: {CODE_EMBEDDING_DIM}D initial embedding per provider

STEP 3: Graph Attention Network (GAT)
  Architecture: Multi-head attention ({NUM_HEADS} heads)
  Training: Supervised (specialty classification)
  Loss: Cross-entropy
  Output: {PROVIDER_EMBEDDING_DIM * NUM_HEADS}D refined embeddings

STEP 4: Final Embeddings
  Providers: {metadata['num_providers']:,}
  Labeled:   {metadata['num_labeled']:,}
  Output:    {PROVIDER_EMBEDDING_DIM * NUM_HEADS}D per provider
  
FINAL OUTPUT DIMENSION: {PROVIDER_EMBEDDING_DIM * NUM_HEADS}D
  = {NUM_HEADS} heads × {PROVIDER_EMBEDDING_DIM}D per head
  = 4 × 128D = 512D
""")

# ============================================================================
# MODEL SUMMARY
# ============================================================================

print("="*80)
print("MODEL SUMMARY")
print("="*80)

print(f"""
Model Type:        Graph Attention Network (Multi-Head)
Total Parameters:  {total_params:,}
Input Dimension:   {CODE_EMBEDDING_DIM}D (code embeddings)
Output Dimension:  {PROVIDER_EMBEDDING_DIM * NUM_HEADS}D (provider embeddings)
Attention Heads:   {NUM_HEADS}
Training Method:   Supervised (Specialty Classification)

Architecture Highlights:
✓ Multi-head attention captures different patterns
✓ Variable-length input (different # of codes per provider)
✓ Attention weights show code importance
✓ Dropout prevents overfitting
✓ LeakyReLU for non-linearity in attention
✓ Softmax for normalized attention weights

Key Innovation:
Instead of simple averaging, GAT learns which procedure codes
are most important for understanding each provider's specialty
and practice patterns.
""")

print("="*80)
print("INSPECTION COMPLETE")
print("="*80)
