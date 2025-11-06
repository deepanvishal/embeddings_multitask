"""
HYBRID PROTOTYPE MODEL: CNP + LEARNED QUERY-DEPENDENT WEIGHTING
================================================================

Combines:
1. Learned query-dependent tower weighting (from labeled data)
2. CNP-based asymmetric similarity (breadth-aware)
3. Real provider prototypes (high in-degree hubs)

Author: AI Assistant
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import random
import time
from sklearn.metrics.pairwise import euclidean_distances

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n" + "="*80)
print("HYBRID PROTOTYPE MODEL TRAINING")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
MARGIN = 0.5
NUM_CNP_PROTOTYPES = 100
CNP_PERCENTILE_NEIGHBORS = 10

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

embeddings_df = pd.read_parquet('final_me2vec_all_towers_1046d.parquet')
all_pins = embeddings_df['PIN'].values
embedding_cols = [col for col in embeddings_df.columns if col != 'PIN']
embeddings = embeddings_df[embedding_cols].values

print(f"Embeddings: {embeddings.shape}")
print(f"Total providers: {len(all_pins)}")

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

print(f"Labeled providers: {len(pin_to_label)}")

pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}
labeled_indices = [pin_to_idx[pin] for pin in pin_to_label.keys()]
labeled_pins = list(pin_to_label.keys())

unique_labels = sorted(set(pin_to_label.values()))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

n_specialties = len(unique_labels)
embedding_dim = embeddings.shape[1]

print(f"Number of specialties: {n_specialties}")
print(f"Embedding dimension: {embedding_dim}")

tower_dims = {
    'procedures': (0, 512),
    'diagnoses': (512, 1024),
    'demographics': (1024, 1029),
    'place': (1029, 1033),
    'cost': (1033, 1044),
    'pin': (1044, 1046)
}

print("\nTower structure:")
for tower, (start, end) in tower_dims.items():
    print(f"  {tower:15s}: dims [{start:4d}:{end:4d}] = {end-start:3d} dims")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

print("\n" + "="*80)
print("DEFINING MODEL ARCHITECTURE")
print("="*80)

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
    
    def get_prototype_similarities(self, query_emb):
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        
        query_norm = F.normalize(query_emb, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        similarities = torch.matmul(query_norm, prototypes_norm.T)
        return similarities.squeeze()

print(f"Model: {n_specialties} specialty prototypes, {embedding_dim}D embeddings")

# ============================================================================
# TOWER WEIGHTING FUNCTIONS
# ============================================================================

def apply_tower_weights_vectorized(embeddings, tower_weights, tower_dims):
    single = embeddings.dim() == 1
    if single:
        embeddings = embeddings.unsqueeze(0)
    
    if tower_weights.dim() == 1:
        tower_weights = tower_weights.unsqueeze(0)
    
    weighted = torch.zeros_like(embeddings)
    
    tower_list = ['procedures', 'diagnoses', 'demographics', 'place', 'cost', 'pin']
    for i, tower_name in enumerate(tower_list):
        start, end = tower_dims[tower_name]
        weighted[:, start:end] = embeddings[:, start:end] * tower_weights[:, i:i+1]
    
    if single:
        weighted = weighted.squeeze(0)
    
    return weighted

# ============================================================================
# TRIPLET LOSS
# ============================================================================

def triplet_loss_vectorized(anchor_embs, positive_embs, negative_embs, tower_weights_batch, tower_dims, margin=0.5):
    weighted_anchors = apply_tower_weights_vectorized(anchor_embs, tower_weights_batch, tower_dims)
    weighted_positives = apply_tower_weights_vectorized(positive_embs, tower_weights_batch, tower_dims)
    weighted_negatives = apply_tower_weights_vectorized(negative_embs, tower_weights_batch, tower_dims)
    
    anchors_norm = F.normalize(weighted_anchors, p=2, dim=1)
    positives_norm = F.normalize(weighted_positives, p=2, dim=1)
    negatives_norm = F.normalize(weighted_negatives, p=2, dim=1)
    
    pos_sim = (anchors_norm * positives_norm).sum(dim=1)
    neg_sim = (anchors_norm * negatives_norm).sum(dim=1)
    
    losses = torch.clamp(margin - pos_sim + neg_sim, min=0.0)
    
    return losses.mean(), pos_sim.mean(), neg_sim.mean()

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================

print("\n" + "="*80)
print("PREPARING TRAINING DATA")
print("="*80)

embeddings_tensor = torch.FloatTensor(embeddings).to(device)

tower_dims_tensor = {}
for name, (start, end) in tower_dims.items():
    tower_dims_tensor[name] = (start, end)

label_to_providers = defaultdict(list)
for pin, label in pin_to_label.items():
    label_to_providers[label].append(pin_to_idx[pin])

for label in label_to_providers:
    label_to_providers[label] = np.array(label_to_providers[label])

print("Creating triplet batches...")

def create_triplet_batch(batch_size):
    anchors = []
    positives = []
    negatives = []
    
    for _ in range(batch_size):
        anchor_label = random.choice(unique_labels)
        anchor_idx = random.choice(label_to_providers[anchor_label])
        
        if len(label_to_providers[anchor_label]) > 1:
            positive_idx = random.choice(label_to_providers[anchor_label])
            while positive_idx == anchor_idx:
                positive_idx = random.choice(label_to_providers[anchor_label])
        else:
            positive_idx = anchor_idx
        
        negative_label = random.choice(unique_labels)
        while negative_label == anchor_label:
            negative_label = random.choice(unique_labels)
        negative_idx = random.choice(label_to_providers[negative_label])
        
        anchors.append(anchor_idx)
        positives.append(positive_idx)
        negatives.append(negative_idx)
    
    return anchors, positives, negatives

n_batches_per_epoch = max(1, len(labeled_indices) // BATCH_SIZE)
print(f"Batches per epoch: {n_batches_per_epoch}")

all_batches = []
for _ in range(n_batches_per_epoch * 3):
    all_batches.append(create_triplet_batch(BATCH_SIZE))

print(f"Pre-created {len(all_batches)} batches")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n" + "="*80)
print("TRAINING QUERY-DEPENDENT WEIGHT MODEL")
print("="*80)

model = PrototypeWeightModel(n_specialties, embedding_dim, n_towers=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

history = {
    'train_loss': [],
    'pos_sim': [],
    'neg_sim': []
}

batch_idx = 0
epoch_start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_losses = []
    epoch_pos_sims = []
    epoch_neg_sims = []
    
    epoch_batches = []
    for _ in range(n_batches_per_epoch):
        epoch_batches.append(all_batches[batch_idx % len(all_batches)])
        batch_idx += 1
    
    for anchor_indices, positive_indices, negative_indices in epoch_batches:
        anchor_tensor = torch.LongTensor(anchor_indices).to(device)
        positive_tensor = torch.LongTensor(positive_indices).to(device)
        negative_tensor = torch.LongTensor(negative_indices).to(device)
        
        anchor_embs = embeddings_tensor[anchor_tensor]
        positive_embs = embeddings_tensor[positive_tensor]
        negative_embs = embeddings_tensor[negative_tensor]
        
        tower_weights_batch = model(anchor_embs)
        
        loss, pos_sim, neg_sim = triplet_loss_vectorized(
            anchor_embs, positive_embs, negative_embs,
            tower_weights_batch, tower_dims_tensor, margin=MARGIN
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_losses.append(loss.item())
        epoch_pos_sims.append(pos_sim.item())
        epoch_neg_sims.append(neg_sim.item())
    
    scheduler.step()
    
    avg_loss = np.mean(epoch_losses)
    avg_pos_sim = np.mean(epoch_pos_sims)
    avg_neg_sim = np.mean(epoch_neg_sims)
    
    history['train_loss'].append(avg_loss)
    history['pos_sim'].append(avg_pos_sim)
    history['neg_sim'].append(avg_neg_sim)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        elapsed = time.time() - epoch_start_time
        time_per_epoch = elapsed / (epoch + 1)
        print(f"Epoch {epoch+1:3d}/{EPOCHS}: "
              f"Loss={avg_loss:.4f}, "
              f"Pos_sim={avg_pos_sim:.4f}, "
              f"Neg_sim={avg_neg_sim:.4f}, "
              f"Margin={avg_pos_sim - avg_neg_sim:.4f}, "
              f"Time={time_per_epoch:.2f}s/epoch")

total_training_time = time.time() - epoch_start_time
print(f"\n✓ Training complete!")
print(f"Total time: {total_training_time:.2f} sec ({total_training_time/60:.2f} min)")

# ============================================================================
# COMPUTE WEIGHTED EMBEDDINGS FOR CNP
# ============================================================================

print("\n" + "="*80)
print("COMPUTING WEIGHTED EMBEDDINGS FOR CNP")
print("="*80)

model.eval()

print("Computing query-dependent weights for all providers...")
all_weighted_embeddings = np.zeros_like(embeddings)

with torch.no_grad():
    for i in range(0, len(all_pins), BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, len(all_pins))
        batch_embs = embeddings_tensor[i:end_idx]
        
        tower_weights = model(batch_embs)
        weighted_batch = apply_tower_weights_vectorized(batch_embs, tower_weights, tower_dims_tensor)
        
        all_weighted_embeddings[i:end_idx] = weighted_batch.cpu().numpy()
        
        if (end_idx) % 1000 == 0:
            print(f"  Processed {end_idx}/{len(all_pins)} providers")

print(f"Weighted embeddings shape: {all_weighted_embeddings.shape}")

# ============================================================================
# COMPUTE CNP MATRIX
# ============================================================================

print("\n" + "="*80)
print("COMPUTING CNP MATRIX")
print("="*80)

print("Computing pairwise distances...")
distances = euclidean_distances(all_weighted_embeddings, all_weighted_embeddings)
print(f"Distance matrix shape: {distances.shape}")

print(f"Computing bandwidth (sigma) from {CNP_PERCENTILE_NEIGHBORS}th percentile...")
percentile_distances = []
for i in range(len(all_pins)):
    sorted_dists = np.sort(distances[i])
    percentile_dist = sorted_dists[min(CNP_PERCENTILE_NEIGHBORS, len(sorted_dists)-1)]
    percentile_distances.append(percentile_dist)

sigma = np.median(percentile_distances)
print(f"Bandwidth sigma: {sigma:.4f}")

print("Computing CNP matrix (this may take a while)...")
cnp_start = time.time()

distances_squared = distances ** 2
numerator = np.exp(-distances_squared / (sigma ** 2))

denominator = numerator.sum(axis=1, keepdims=True)
cnp_matrix = numerator / (denominator + 1e-10)

np.fill_diagonal(cnp_matrix, 0)

cnp_time = time.time() - cnp_start
print(f"✓ CNP matrix computed in {cnp_time:.2f} sec")
print(f"CNP matrix shape: {cnp_matrix.shape}")
print(f"CNP matrix stats: min={cnp_matrix.min():.6f}, max={cnp_matrix.max():.6f}, mean={cnp_matrix.mean():.6f}")

# ============================================================================
# SELECT PROTOTYPES VIA IN-DEGREE
# ============================================================================

print("\n" + "="*80)
print("SELECTING PROTOTYPES VIA CNP IN-DEGREE")
print("="*80)

in_degrees = cnp_matrix.sum(axis=0)
print(f"In-degree stats: min={in_degrees.min():.4f}, max={in_degrees.max():.4f}, mean={in_degrees.mean():.4f}")

k_prototypes = min(NUM_CNP_PROTOTYPES, len(all_pins))
prototype_indices = np.argsort(in_degrees)[-k_prototypes:][::-1]
prototype_pins = all_pins[prototype_indices]

print(f"\nSelected {len(prototype_indices)} prototypes")
print(f"Top 10 prototypes by in-degree:")
for i, idx in enumerate(prototype_indices[:10]):
    pin = all_pins[idx]
    in_deg = in_degrees[idx]
    specialty = pin_to_label.get(pin, 'UNLABELED')
    print(f"  {i+1:2d}. PIN {pin:12s} In-degree={in_deg:8.4f} Specialty={specialty}")

# ============================================================================
# ASSIGN PROVIDERS TO PROTOTYPES
# ============================================================================

print("\n" + "="*80)
print("ASSIGNING PROVIDERS TO PROTOTYPES")
print("="*80)

provider_to_prototype = {}

print("Computing CNP from each provider to prototypes...")
for i, pin in enumerate(all_pins):
    cnp_to_prototypes = cnp_matrix[i, prototype_indices]
    nearest_prototype_idx = prototype_indices[np.argmax(cnp_to_prototypes)]
    nearest_prototype_pin = all_pins[nearest_prototype_idx]
    
    provider_to_prototype[pin] = {
        'prototype_pin': nearest_prototype_pin,
        'prototype_idx': nearest_prototype_idx,
        'cnp_score': cnp_to_prototypes.max()
    }
    
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1}/{len(all_pins)} providers")

print(f"✓ All {len(all_pins)} providers assigned to prototypes")

prototype_cluster_sizes = defaultdict(int)
for assignment in provider_to_prototype.values():
    prototype_cluster_sizes[assignment['prototype_pin']] += 1

print(f"\nPrototype cluster sizes:")
print(f"  Min: {min(prototype_cluster_sizes.values())}")
print(f"  Max: {max(prototype_cluster_sizes.values())}")
print(f"  Mean: {np.mean(list(prototype_cluster_sizes.values())):.1f}")

# ============================================================================
# SAVE EVERYTHING
# ============================================================================

print("\n" + "="*80)
print("SAVING MODEL AND RESULTS")
print("="*80)

torch.save({
    'model_state_dict': model.state_dict(),
    'n_specialties': n_specialties,
    'embedding_dim': embedding_dim,
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'tower_dims': tower_dims,
    'history': history
}, 'trained_prototype_model.pth')
print("✓ Saved: trained_prototype_model.pth")

with open('cnp_prototypes.pkl', 'wb') as f:
    pickle.dump({
        'prototype_pins': prototype_pins.tolist(),
        'prototype_indices': prototype_indices.tolist(),
        'in_degrees': in_degrees[prototype_indices].tolist(),
        'sigma': sigma,
        'k_prototypes': k_prototypes
    }, f)
print("✓ Saved: cnp_prototypes.pkl")

with open('provider_to_prototype.pkl', 'wb') as f:
    pickle.dump(provider_to_prototype, f)
print("✓ Saved: provider_to_prototype.pkl")

np.save('cnp_in_degrees.npy', in_degrees)
print("✓ Saved: cnp_in_degrees.npy")

np.save('weighted_embeddings.npy', all_weighted_embeddings)
print("✓ Saved: weighted_embeddings.npy")

with open('hybrid_model_metadata.pkl', 'wb') as f:
    pickle.dump({
        'unique_labels': unique_labels,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'tower_dims': tower_dims,
        'n_specialties': n_specialties,
        'embedding_dim': embedding_dim,
        'k_prototypes': k_prototypes,
        'sigma': sigma,
        'total_providers': len(all_pins),
        'labeled_providers': len(pin_to_label),
        'training_time': total_training_time,
        'cnp_computation_time': cnp_time,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'margin': MARGIN
    }, f)
print("✓ Saved: hybrid_model_metadata.pkl")

# ============================================================================
# VALIDATION EXAMPLE
# ============================================================================

print("\n" + "="*80)
print("VALIDATION EXAMPLE")
print("="*80)

if len(pin_to_label) > 0:
    query_pin = random.choice(list(pin_to_label.keys()))
    query_idx = pin_to_idx[query_pin]
    query_emb = embeddings_tensor[query_idx]
    true_label = pin_to_label[query_pin]
    
    print(f"\nQuery provider: PIN {query_pin}")
    print(f"True specialty: {true_label}")
    
    with torch.no_grad():
        predicted_weights = model(query_emb)
    
    print(f"\nPredicted tower weights:")
    tower_names = ['Procedures', 'Diagnoses', 'Demographics', 'Place', 'Cost', 'PIN']
    for i, name in enumerate(tower_names):
        print(f"  {name:15s}: {predicted_weights[i].item():.4f}")
    
    cnp_to_prototypes = cnp_matrix[query_idx, prototype_indices]
    top_5_proto_idx = np.argsort(cnp_to_prototypes)[-5:][::-1]
    
    print(f"\nTop 5 prototypes by CNP:")
    for rank, proto_idx in enumerate(top_5_proto_idx, 1):
        proto_pin = prototype_pins[proto_idx]
        proto_specialty = pin_to_label.get(proto_pin, 'UNLABELED')
        cnp_score = cnp_to_prototypes[proto_idx]
        print(f"  {rank}. PIN {proto_pin:12s} CNP={cnp_score:.4f} Specialty={proto_specialty}")
    
    assignment = provider_to_prototype[query_pin]
    print(f"\nAssigned to prototype: {assignment['prototype_pin']}")
    print(f"CNP score: {assignment['cnp_score']:.4f}")

print("\n" + "="*80)
print("HYBRID PROTOTYPE MODEL TRAINING COMPLETE")
print("="*80)
print(f"\nTraining time: {total_training_time:.2f} sec ({total_training_time/60:.2f} min)")
print(f"CNP computation time: {cnp_time:.2f} sec")
print(f"\nSaved files:")
print("  - trained_prototype_model.pth")
print("  - cnp_prototypes.pkl")
print("  - provider_to_prototype.pkl")
print("  - cnp_in_degrees.npy")
print("  - weighted_embeddings.npy")
print("  - hybrid_model_metadata.pkl")
print("\nReady for inference!")
