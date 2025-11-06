"""
INSPECT MODEL ARCHITECTURE
===========================
Load trained model and display architecture details for SME review.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

print("="*80)
print("LOADING TRAINED MODEL")
print("="*80)

# ============================================================================
# DEFINE MODEL ARCHITECTURE
# ============================================================================

class PrototypeWeightModel(nn.Module):
    def __init__(self, n_specialties, embedding_dim, n_towers=6):
        super().__init__()
        
        self.n_specialties = n_specialties
        self.embedding_dim = embedding_dim
        self.n_towers = n_towers
        
        self.prototypes = nn.Parameter(torch.randn(n_specialties, embedding_dim) * 0.1)
        self.weight_profiles = nn.Parameter(torch.ones(n_specialties, n_towers))
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, query_emb):
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        query_norm = F.normalize(query_emb, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        similarities = torch.matmul(query_norm, prototypes_norm.T)
        similarities = similarities / self.temperature
        prototype_weights = F.softmax(similarities, dim=1)
        
        tower_weights = torch.matmul(prototype_weights, self.weight_profiles)
        tower_weights = F.softmax(tower_weights, dim=1)
        
        if squeeze_output:
            tower_weights = tower_weights.squeeze(0)
        
        return tower_weights

# ============================================================================
# LOAD MODEL
# ============================================================================

checkpoint = torch.load('trained_prototype_model.pth', map_location='cpu')

model = PrototypeWeightModel(
    n_specialties=checkpoint['n_specialties'],
    embedding_dim=checkpoint['embedding_dim'],
    n_towers=6
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded successfully!\n")

# ============================================================================
# DISPLAY MODEL ARCHITECTURE
# ============================================================================

print("="*80)
print("MODEL ARCHITECTURE SUMMARY")
print("="*80)

print("\n" + "-"*80)
print("MODEL CLASS: PrototypeWeightModel")
print("-"*80)

print(f"\nHyperparameters:")
print(f"  Number of Specialties:  {model.n_specialties}")
print(f"  Embedding Dimension:    {model.embedding_dim}")
print(f"  Number of Towers:       {model.n_towers}")

print("\n" + "-"*80)
print("LEARNABLE PARAMETERS")
print("-"*80)

total_params = 0

for name, param in model.named_parameters():
    num_params = param.numel()
    total_params += num_params
    print(f"\n{name}:")
    print(f"  Shape:      {list(param.shape)}")
    print(f"  Parameters: {num_params:,}")
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

print("""
INPUT:
  Query Embedding [1 × 1046]
  
LAYER 1: Normalize Query & Prototypes
  query_norm = F.normalize(query_emb)          → [1 × 1046]
  prototypes_norm = F.normalize(prototypes)     → [50 × 1046]
  
LAYER 2: Compute Similarity
  similarities = query_norm @ prototypes_norm.T → [1 × 50]
  
LAYER 3: Temperature Scaling
  scaled = similarities / temperature            → [1 × 50]
  
LAYER 4: Softmax (Prototype Attention)
  prototype_weights = softmax(scaled)            → [1 × 50]
  
LAYER 5: Weight Profile Blending
  tower_weights = prototype_weights @ weight_profiles → [1 × 6]
  
LAYER 6: Final Softmax (Normalize Tower Weights)
  output = softmax(tower_weights)                → [1 × 6]
  
OUTPUT:
  Tower Weights [6] - importance for each tower
""")

# ============================================================================
# DISPLAY TOWER DIMENSIONS
# ============================================================================

print("="*80)
print("TOWER STRUCTURE")
print("="*80)

tower_dims = checkpoint['tower_dims']

print(f"\n{'Tower Name':<20s} {'Start':<8s} {'End':<8s} {'Dimensions':<12s}")
print("-"*80)
for tower_name, (start, end) in tower_dims.items():
    dim_size = end - start
    print(f"{tower_name:<20s} {start:<8d} {end:<8d} {dim_size:<12d}")

print(f"\nTotal Embedding Dimension: {checkpoint['embedding_dim']}")

# ============================================================================
# DISPLAY LEARNED WEIGHT PROFILES
# ============================================================================

print("\n" + "="*80)
print("LEARNED WEIGHT PROFILES (Per Specialty)")
print("="*80)

weight_profiles = model.weight_profiles.data.cpu().numpy()

with open('hybrid_model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

idx_to_label = metadata['idx_to_label']

print(f"\n{'Specialty':<30s} {'Proc':>8s} {'Diag':>8s} {'Demo':>8s} {'Place':>8s} {'Cost':>8s} {'PIN':>8s}")
print("-"*80)

for idx in range(min(10, model.n_specialties)):
    if idx in idx_to_label:
        specialty = idx_to_label[idx]
        weights = weight_profiles[idx]
        weights_softmax = torch.softmax(torch.tensor(weights), dim=0).numpy()
        
        print(f"{specialty:<30s} {weights_softmax[0]:>8.3f} {weights_softmax[1]:>8.3f} "
              f"{weights_softmax[2]:>8.3f} {weights_softmax[3]:>8.3f} "
              f"{weights_softmax[4]:>8.3f} {weights_softmax[5]:>8.3f}")

print(f"\n(Showing first 10 of {model.n_specialties} specialties)")

# ============================================================================
# DISPLAY CURRENT TEMPERATURE
# ============================================================================

print("\n" + "="*80)
print("LEARNED TEMPERATURE PARAMETER")
print("="*80)

temperature = model.temperature.item()
print(f"\nTemperature: {temperature:.4f}")
print(f"\nInterpretation:")
if temperature < 0.5:
    print("  → SHARP attention (strongly focuses on most similar prototype)")
elif temperature > 2.0:
    print("  → SOFT attention (blends many prototypes)")
else:
    print("  → BALANCED attention (moderate blending)")

# ============================================================================
# TEST FORWARD PASS
# ============================================================================

print("\n" + "="*80)
print("TEST FORWARD PASS")
print("="*80)

dummy_embedding = torch.randn(1, checkpoint['embedding_dim'])

print(f"\nInput shape: {list(dummy_embedding.shape)}")

with torch.no_grad():
    output = model(dummy_embedding)

print(f"Output shape: {list(output.shape)}")
print(f"\nOutput tower weights:")
tower_names = ['Procedures', 'Diagnoses', 'Demographics', 'Place', 'Cost', 'PIN']
for i, name in enumerate(tower_names):
    print(f"  {name:<15s}: {output[i].item():.4f}")

print(f"\nSum of weights: {output.sum().item():.4f} (should be 1.0)")

# ============================================================================
# TRAINING HISTORY
# ============================================================================

if 'history' in checkpoint:
    print("\n" + "="*80)
    print("TRAINING HISTORY")
    print("="*80)
    
    history = checkpoint['history']
    
    print(f"\nFinal Training Metrics:")
    print(f"  Train Loss:        {history['train_loss'][-1]:.4f}")
    print(f"  Positive Sim:      {history['pos_sim'][-1]:.4f}")
    print(f"  Negative Sim:      {history['neg_sim'][-1]:.4f}")
    print(f"  Margin (Pos-Neg):  {history['pos_sim'][-1] - history['neg_sim'][-1]:.4f}")
    
    print(f"\nTraining Progress:")
    print(f"  Initial Loss: {history['train_loss'][0]:.4f}")
    print(f"  Final Loss:   {history['train_loss'][-1]:.4f}")
    print(f"  Improvement:  {history['train_loss'][0] - history['train_loss'][-1]:.4f}")

# ============================================================================
# MODEL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)

print(f"""
Model Type:        Prototype-Based Query-Dependent Tower Weighting
Total Parameters:  {total_params:,}
Specialties:       {model.n_specialties}
Input Dimension:   {model.embedding_dim}
Output Dimension:  {model.n_towers}
Training Method:   Metric Learning (Triplet Loss)

Architecture Flow:
1. Query embedding → Similarity to specialty prototypes
2. Temperature-scaled softmax → Attention weights
3. Blend weight profiles → Tower importance
4. Softmax normalization → Final tower weights

Key Features:
✓ Query-dependent tower weighting
✓ Specialty-specific weight profiles
✓ Learnable temperature for attention sharpness
✓ Fully differentiable architecture
✓ ~52K parameters for 50 specialties
""")

print("="*80)
print("INSPECTION COMPLETE")
print("="*80)
