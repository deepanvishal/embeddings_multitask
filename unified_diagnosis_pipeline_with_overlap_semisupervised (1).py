import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import json
from scipy.sparse import load_npz
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================

MIN_PARAMETER_RATIO = 10
OVERFITTING_THRESHOLD = 1.5
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
BATCH_SIZE = 64
BATCH_SIZE_INFERENCE = 256
OVERLAP_EMBEDDING_DIM = 32

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")
diag_matrix = load_npz('diagnosis_vectors.npz')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

with open('specialty_code_mappings.pkl', 'rb') as f:
    specialty_mappings = pickle.load(f)

specialty_code_indices = specialty_mappings['code_indices']

print(f"Total providers: {len(all_pins)}")
print(f"Labeled providers: {len(pin_to_label)}")
print(f"Unlabeled providers: {len(all_pins) - len(pin_to_label)}")
print(f"Total diagnosis codes: {diag_matrix.shape[1]}")

print("\nFiltering to specialty-relevant codes...")
all_specialty_codes = set()
for spec, code_indices in specialty_code_indices.items():
    all_specialty_codes.update(code_indices)
all_specialty_codes = sorted(list(all_specialty_codes))

print(f"Specialty-relevant codes: {len(all_specialty_codes)}")

diag_matrix_filtered = diag_matrix[:, all_specialty_codes]

labeled_mask = [pin in pin_to_label for pin in all_pins]
labeled_indices = [i for i, is_labeled in enumerate(labeled_mask) if is_labeled]
diag_matrix_labeled = diag_matrix_filtered[labeled_indices]
all_pins_labeled = [all_pins[i] for i in labeled_indices]

diag_tensor_labeled = torch.FloatTensor(diag_matrix_labeled.toarray()).to(device)
diag_tensor_full = torch.FloatTensor(diag_matrix_filtered.toarray()).to(device)

input_dim = diag_tensor_labeled.shape[1]
n_labeled = len(all_pins_labeled)
n_total = len(all_pins)

print(f"Input dimension: {input_dim}")

unique_specialties = sorted(set(pin_to_label.values()))
specialty_to_id = {spec: idx for idx, spec in enumerate(unique_specialties)}
id_to_specialty = {idx: spec for spec, idx in specialty_to_id.items()}
num_specialties = len(unique_specialties)

pin_to_idx_labeled = {pin: idx for idx, pin in enumerate(all_pins_labeled)}
specialty_ids_labeled = torch.LongTensor([
    specialty_to_id[pin_to_label[pin]] for pin in all_pins_labeled
]).to(device)

print(f"Number of specialties: {num_specialties}")

specialty_counts = defaultdict(int)
for specialty in pin_to_label.values():
    specialty_counts[specialty] += 1

print("\nSpecialty distribution:")
for spec in sorted(specialty_counts.keys(), key=lambda x: specialty_counts[x], reverse=True):
    print(f"  {spec:25s}: {specialty_counts[spec]:4d}")

np.random.seed(42)
indices = list(range(n_labeled))
np.random.shuffle(indices)

n_train = int(0.7 * n_labeled)
n_val = int(0.15 * n_labeled)

train_indices = indices[:n_train]
val_indices = indices[n_train:n_train+n_val]
test_indices = indices[n_train+n_val:]

print(f"\nTrain: {len(train_indices)} ({100*len(train_indices)/n_labeled:.1f}%)")
print(f"Val:   {len(val_indices)} ({100*len(val_indices)/n_labeled:.1f}%)")
print(f"Test:  {len(test_indices)} ({100*len(test_indices)/n_labeled:.1f}%)")

total_samples = len(pin_to_label)
specialty_weights = {}
for specialty, count in specialty_counts.items():
    spec_id = specialty_to_id[specialty]
    weight = total_samples / (num_specialties * count)
    specialty_weights[spec_id] = weight

min_weight = min(specialty_weights.values())
max_weight = 10 * min_weight

for spec_id in specialty_weights:
    specialty_weights[spec_id] = min(specialty_weights[spec_id], max_weight)

weight_tensor = torch.FloatTensor([specialty_weights[i] for i in range(num_specialties)]).to(device)

small_specialties = [spec for spec, count in specialty_counts.items() if count < 100]
small_specialty_ids = [specialty_to_id[spec] for spec in small_specialties]

stage1_train_indices = []
for spec_id in small_specialty_ids:
    spec_indices = [idx for idx in train_indices if specialty_ids_labeled[idx] == spec_id]
    stage1_train_indices.extend(spec_indices)

large_specialty_ids = [sid for sid in range(num_specialties) if sid not in small_specialty_ids]
for spec_id in large_specialty_ids:
    spec_indices = [idx for idx in train_indices if specialty_ids_labeled[idx] == spec_id]
    sample_size = int(0.2 * len(spec_indices))
    stage1_train_indices.extend(np.random.choice(spec_indices, sample_size, replace=False).tolist())

print(f"\nStage 1 training samples: {len(stage1_train_indices)}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_model_parameters(input_dim, latent_dim, specialty_emb_dim=32):
    enc_params = (input_dim + specialty_emb_dim) * 512 + 512 * 256 + 256 * latent_dim
    dec_params = (latent_dim + specialty_emb_dim) * 256 + 256 * 512 + 512 * input_dim
    specialty_emb_params = num_specialties * specialty_emb_dim
    total = enc_params + dec_params + specialty_emb_params
    return total

def check_overfitting(train_loss, val_loss, threshold=OVERFITTING_THRESHOLD):
    ratio = val_loss / train_loss
    if ratio > threshold:
        return True, ratio
    return False, ratio

def masked_mse_loss(recon, x):
    mask = (x > 0).float()
    mse = ((recon - x) ** 2) * mask
    loss = mse.sum() / (mask.sum() + 1e-8)
    return loss

def weighted_loss(recon, x, specialty_ids, weight_tensor):
    sample_losses = F.mse_loss(recon, x, reduction='none').mean(dim=1)
    batch_weights = weight_tensor[specialty_ids]
    return (sample_losses * batch_weights).mean()

class BalancedSpecialtyBatchSampler:
    def __init__(self, indices, specialty_ids_tensor, num_specialties, batch_size):
        self.indices = indices
        self.specialty_ids = specialty_ids_tensor
        self.num_specialties = num_specialties
        self.batch_size = batch_size
        self.samples_per_specialty = max(1, batch_size // num_specialties)
        
        self.specialty_pools = defaultdict(list)
        for idx in indices:
            spec_id = int(self.specialty_ids[idx].item())
            self.specialty_pools[spec_id].append(idx)
        
        for spec_id in self.specialty_pools:
            np.random.shuffle(self.specialty_pools[spec_id])
        
        self.iterators = {spec_id: 0 for spec_id in range(num_specialties)}
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch = []
        for spec_id in range(self.num_specialties):
            if spec_id not in self.specialty_pools or len(self.specialty_pools[spec_id]) == 0:
                continue
            
            pool = self.specialty_pools[spec_id]
            start = self.iterators[spec_id]
            
            for _ in range(self.samples_per_specialty):
                if start >= len(pool):
                    np.random.shuffle(pool)
                    start = 0
                batch.append(pool[start])
                start += 1
            
            self.iterators[spec_id] = start
        
        if len(batch) == 0:
            raise StopIteration
        
        np.random.shuffle(batch)
        return batch

# ============================================================================
# PHASE 1: SPECIALTY-CONDITIONED AUTOENCODER
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: LATENT DIMENSION SEARCH")
print("="*80)

class SpecialtyConditionedAutoencoder(nn.Module):
    def __init__(self, input_dim, num_specialties, latent_dim, specialty_emb_dim=32):
        super().__init__()
        
        self.specialty_embedding = nn.Embedding(num_specialties, specialty_emb_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + specialty_emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + specialty_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, input_dim)
        )
    
    def encode(self, x, specialty_id):
        spec_emb = self.specialty_embedding(specialty_id)
        encoder_input = torch.cat([x, spec_emb], dim=1)
        return self.encoder(encoder_input)
    
    def decode(self, z, specialty_id):
        spec_emb = self.specialty_embedding(specialty_id)
        decoder_input = torch.cat([z, spec_emb], dim=1)
        return self.decoder(decoder_input)
    
    def forward(self, x, specialty_id):
        z = self.encode(x, specialty_id)
        recon = self.decode(z, specialty_id)
        return recon, z

latent_dim_candidates = [32, 64, 128, 256]
safe_latent_dims = []

print(f"\nChecking parameter ratios (min required: 1:{MIN_PARAMETER_RATIO}):")
for latent_dim in latent_dim_candidates:
    params = calculate_model_parameters(input_dim, latent_dim)
    ratio = params / len(train_indices)
    
    print(f"  Latent dim {latent_dim:3d}: {params:,} params, ratio 1:{ratio:.1f}", end="")
    
    if ratio <= MIN_PARAMETER_RATIO:
        safe_latent_dims.append(latent_dim)
        print(" [SAFE]")
    else:
        print(" [RISKY - SKIPPED]")

if len(safe_latent_dims) == 0:
    print("\nWARNING: No safe latent dimensions found.")
    print("Using smallest option (32) with extreme regularization.")
    safe_latent_dims = [32]

print(f"\nTesting latent dimensions: {safe_latent_dims}")

def quick_eval_latent_dim(latent_dim, epochs=20):
    model = SpecialtyConditionedAutoencoder(input_dim, num_specialties, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        
        n_batches = len(train_indices) // BATCH_SIZE
        for i in range(n_batches):
            batch_idx = train_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_x = diag_tensor_labeled[batch_idx]
            batch_spec = specialty_ids_labeled[batch_idx]
            
            recon, z = model(batch_x, batch_spec)
            loss = masked_mse_loss(recon, batch_x)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_batch_losses = []
            n_val_batches = len(val_indices) // BATCH_SIZE
            for i in range(n_val_batches):
                batch_idx = val_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                batch_x = diag_tensor_labeled[batch_idx]
                batch_spec = specialty_ids_labeled[batch_idx]
                
                recon, z = model(batch_x, batch_spec)
                loss = masked_mse_loss(recon, batch_x)
                val_batch_losses.append(loss.item())
        
        train_loss = np.mean(batch_losses)
        val_loss = np.mean(val_batch_losses)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    best_val = min(val_losses)
    overfit_ratio = final_val / final_train
    
    return {
        'latent_dim': latent_dim,
        'final_train': final_train,
        'final_val': final_val,
        'best_val': best_val,
        'overfit_ratio': overfit_ratio
    }

search_results = []
for latent_dim in safe_latent_dims:
    print(f"\nEvaluating latent_dim={latent_dim}...")
    result = quick_eval_latent_dim(latent_dim, epochs=20)
    search_results.append(result)
    
    is_overfitting, ratio = check_overfitting(result['final_train'], result['final_val'])
    status = "OVERFITTING WARNING" if is_overfitting else "OK"
    
    print(f"  Train: {result['final_train']:.4f}")
    print(f"  Val:   {result['final_val']:.4f}")
    print(f"  Ratio: {ratio:.3f} [{status}]")

valid_results = [r for r in search_results if r['overfit_ratio'] < OVERFITTING_THRESHOLD]
if len(valid_results) == 0:
    print("\nWARNING: All configurations show overfitting. Using lowest val loss.")
    best_result = min(search_results, key=lambda x: x['final_val'])
else:
    best_result = min(valid_results, key=lambda x: x['final_val'])

PHASE1_LATENT_DIM = best_result['latent_dim']

print(f"\n" + "="*80)
print(f"SELECTED LATENT DIM: {PHASE1_LATENT_DIM}")
print(f"  Val loss: {best_result['final_val']:.4f}")
print(f"  Overfit ratio: {best_result['overfit_ratio']:.3f}")
print("="*80)

with open('phase1_latent_dim_search.json', 'w') as f:
    json.dump(search_results, f, indent=2)

# ============================================================================
# PHASE 1: FULL TRAINING
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: TRAINING SPECIALTY-CONDITIONED AUTOENCODER")
print("="*80)
print(f"Latent dimension: {PHASE1_LATENT_DIM}")

model = SpecialtyConditionedAutoencoder(input_dim, num_specialties, PHASE1_LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float('inf')
patience_counter = 0
history = {'stage1_train': [], 'stage1_val': [], 'stage2_train': [], 'stage2_val': []}

print("\nStage 1: Small specialty focus (30 epochs)")
for epoch in range(30):
    model.train()
    train_losses = []
    
    np.random.shuffle(stage1_train_indices)
    n_batches = len(stage1_train_indices) // BATCH_SIZE
    
    for i in range(n_batches):
        batch_idx = stage1_train_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_x = diag_tensor_labeled[batch_idx]
        batch_spec = specialty_ids_labeled[batch_idx]
        
        recon, z = model(batch_x, batch_spec)
        loss = weighted_loss(recon, batch_x, batch_spec, weight_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        val_batch_losses = []
        n_val_batches = len(val_indices) // BATCH_SIZE
        for i in range(n_val_batches):
            batch_idx = val_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_x = diag_tensor_labeled[batch_idx]
            batch_spec = specialty_ids_labeled[batch_idx]
            
            recon, z = model(batch_x, batch_spec)
            loss = masked_mse_loss(recon, batch_x)
            val_batch_losses.append(loss.item())
    
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_batch_losses)
    
    history['stage1_train'].append(train_loss)
    history['stage1_val'].append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    is_overfitting, ratio = check_overfitting(train_loss, val_loss)
    status = " [OVERFITTING WARNING]" if is_overfitting else ""
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"Best={best_val_loss:.4f}, Ratio={ratio:.3f}{status}")
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_model_state)

print("\nStage 2: Balanced training (70 epochs)")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.5)

balanced_sampler = BalancedSpecialtyBatchSampler(
    train_indices, specialty_ids_labeled, num_specialties, BATCH_SIZE
)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(70):
    model.train()
    train_losses = []
    
    sampler_iter = iter(balanced_sampler)
    for _ in range(len(train_indices) // BATCH_SIZE):
        try:
            batch_idx = next(sampler_iter)
        except StopIteration:
            break
        
        batch_x = diag_tensor_labeled[batch_idx]
        batch_spec = specialty_ids_labeled[batch_idx]
        
        recon, z = model(batch_x, batch_spec)
        loss = weighted_loss(recon, batch_x, batch_spec, weight_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        val_batch_losses = []
        n_val_batches = len(val_indices) // BATCH_SIZE
        for i in range(n_val_batches):
            batch_idx = val_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_x = diag_tensor_labeled[batch_idx]
            batch_spec = specialty_ids_labeled[batch_idx]
            
            recon, z = model(batch_x, batch_spec)
            loss = masked_mse_loss(recon, batch_x)
            val_batch_losses.append(loss.item())
    
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_batch_losses)
    
    history['stage2_train'].append(train_loss)
    history['stage2_val'].append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    is_overfitting, ratio = check_overfitting(train_loss, val_loss)
    status = " [OVERFITTING WARNING]" if is_overfitting else ""
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"Best={best_val_loss:.4f}, Ratio={ratio:.3f}{status}")
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_model_state)

model.eval()
with torch.no_grad():
    test_batch_losses = []
    n_test_batches = len(test_indices) // BATCH_SIZE
    for i in range(n_test_batches):
        batch_idx = test_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_x = diag_tensor_labeled[batch_idx]
        batch_spec = specialty_ids_labeled[batch_idx]
        
        recon, z = model(batch_x, batch_spec)
        loss = masked_mse_loss(recon, batch_x)
        test_batch_losses.append(loss.item())
    
    test_loss = np.mean(test_batch_losses)

final_train = history['stage2_train'][-1]
final_val = history['stage2_val'][-1]

print(f"\nPhase 1 Complete:")
print(f"  Train loss: {final_train:.4f}")
print(f"  Val loss:   {final_val:.4f}")
print(f"  Test loss:  {test_loss:.4f}")

is_overfitting_val, phase1_val_ratio = check_overfitting(final_train, final_val)
is_overfitting_test, phase1_test_ratio = check_overfitting(final_train, test_loss)

print(f"\nOverfitting check:")
print(f"  Val/Train:  {phase1_val_ratio:.3f} {'[OVERFITTING WARNING]' if is_overfitting_val else '[OK]'}")
print(f"  Test/Train: {phase1_test_ratio:.3f} {'[OVERFITTING WARNING]' if is_overfitting_test else '[OK]'}")

torch.save(model.state_dict(), 'phase1_diagnosis_specialty_conditioned_autoencoder.pth')

metadata = {
    'latent_dim': PHASE1_LATENT_DIM,
    'input_dim': input_dim,
    'num_specialties': num_specialties,
    'specialty_to_id': specialty_to_id,
    'id_to_specialty': id_to_specialty,
    'specialty_code_indices': all_specialty_codes,
    'original_diagnosis_dim': diag_matrix.shape[1],
    'final_train_loss': final_train,
    'final_val_loss': final_val,
    'final_test_loss': test_loss,
    'val_train_ratio': phase1_val_ratio,
    'test_train_ratio': phase1_test_ratio,
    'latent_dim_search': search_results
}

with open('phase1_diagnosis_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\nSaved: phase1_diagnosis_specialty_conditioned_autoencoder.pth")
print("Saved: phase1_diagnosis_metadata.pkl")

# ============================================================================
# PHASE 2: GENERATE MULTI-VIEW EMBEDDINGS
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: GENERATING MULTI-VIEW EMBEDDINGS")
print("="*80)

multiview_dim = num_specialties * PHASE1_LATENT_DIM
print(f"Multi-view dimension: {num_specialties} specialties × {PHASE1_LATENT_DIM} = {multiview_dim}")

model.eval()
all_multiview = []

with torch.no_grad():
    for i in range(0, n_total, BATCH_SIZE_INFERENCE):
        end_idx = min(i + BATCH_SIZE_INFERENCE, n_total)
        batch_x = diag_tensor_full[i:end_idx]
        
        batch_multiview = []
        for spec_id in range(num_specialties):
            spec_ids = torch.full((end_idx - i,), spec_id, dtype=torch.long, device=device)
            embeddings = model.encode(batch_x, spec_ids)
            batch_multiview.append(embeddings.cpu().numpy())
        
        combined = np.concatenate(batch_multiview, axis=1)
        all_multiview.append(combined)
        
        if (i // BATCH_SIZE_INFERENCE) % 10 == 0:
            print(f"  Encoded {end_idx}/{n_total} providers...")

multiview_embeddings = np.vstack(all_multiview)
print(f"\nMulti-view embeddings shape: {multiview_embeddings.shape}")

np.save('phase2_diagnosis_multiview_embeddings.npy', multiview_embeddings)
print("Saved: phase2_diagnosis_multiview_embeddings.npy")

# ============================================================================
# PHASE 2.5: OVERLAP AUTOENCODER
# ============================================================================

print("\n" + "="*80)
print("PHASE 2.5: TRAINING OVERLAP AUTOENCODER")
print("="*80)

class OverlapAutoencoder(nn.Module):
    def __init__(self, input_dim, overlap_dim=32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, overlap_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(overlap_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return recon, z

print(f"Input dimension: {input_dim}")
print(f"Overlap embedding dimension: {OVERLAP_EMBEDDING_DIM}")

overlap_params = (input_dim * 256 + 256 * 128 + 128 * OVERLAP_EMBEDDING_DIM) * 2
print(f"Parameters: {overlap_params:,}")

n_overlap_train = int(0.8 * n_total)
overlap_train_indices = list(range(n_overlap_train))
overlap_val_indices = list(range(n_overlap_train, n_total))

print(f"Training on ALL {n_total} providers (unsupervised)")
print(f"  Train: {len(overlap_train_indices)}")
print(f"  Val:   {len(overlap_val_indices)}")

overlap_model = OverlapAutoencoder(input_dim, OVERLAP_EMBEDDING_DIM).to(device)
optimizer = torch.optim.Adam(overlap_model.parameters(), lr=LEARNING_RATE)

best_val_loss = float('inf')
patience_counter = 0
overlap_history = {'train': [], 'val': []}

for epoch in range(50):
    overlap_model.train()
    train_losses = []
    
    n_batches = len(overlap_train_indices) // BATCH_SIZE
    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(overlap_train_indices))
        batch_idx = overlap_train_indices[start:end]
        batch_x = diag_tensor_full[batch_idx]
        
        recon, z = overlap_model(batch_x)
        loss = F.mse_loss(recon, batch_x)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(overlap_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
    
    overlap_model.eval()
    with torch.no_grad():
        val_batch_losses = []
        n_val_batches = len(overlap_val_indices) // BATCH_SIZE
        for i in range(n_val_batches):
            start = i * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(overlap_val_indices))
            batch_idx = overlap_val_indices[start:end]
            batch_x = diag_tensor_full[batch_idx]
            
            recon, z = overlap_model(batch_x)
            loss = F.mse_loss(recon, batch_x)
            val_batch_losses.append(loss.item())
    
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_batch_losses)
    
    overlap_history['train'].append(train_loss)
    overlap_history['val'].append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_overlap_state = overlap_model.state_dict().copy()
    else:
        patience_counter += 1
    
    is_overfitting, ratio = check_overfitting(train_loss, val_loss)
    status = " [OVERFITTING WARNING]" if is_overfitting else ""
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"Best={best_val_loss:.4f}, Ratio={ratio:.3f}{status}")
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

overlap_model.load_state_dict(best_overlap_state)

print(f"\nPhase 2.5 Complete:")
print(f"  Best val loss: {best_val_loss:.4f}")

torch.save(overlap_model.state_dict(), 'overlap_diagnosis_autoencoder.pth')
print("Saved: overlap_diagnosis_autoencoder.pth")

overlap_model.eval()
overlap_embeddings = []

with torch.no_grad():
    for i in range(0, n_total, BATCH_SIZE_INFERENCE):
        end_idx = min(i + BATCH_SIZE_INFERENCE, n_total)
        batch_x = diag_tensor_full[i:end_idx]
        z = overlap_model.encode(batch_x)
        overlap_embeddings.append(z.cpu().numpy())

overlap_embeddings = np.vstack(overlap_embeddings)
print(f"Overlap embeddings shape: {overlap_embeddings.shape}")

np.save('overlap_diagnosis_embeddings_32d.npy', overlap_embeddings)
print("Saved: overlap_diagnosis_embeddings_32d.npy")

# ============================================================================
# PHASE 3: SEMI-SUPERVISED COMPRESSION
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: SEMI-SUPERVISED COMPRESSION NETWORK")
print("="*80)

FINAL_DIM = 128
combined_input_dim = multiview_dim + OVERLAP_EMBEDDING_DIM

print(f"Input: multiview ({multiview_dim}) + overlap ({OVERLAP_EMBEDDING_DIM}) = {combined_input_dim}")
print(f"Output: {FINAL_DIM}")

class CompressionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        return self.network(x)

def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    device = embeddings.device
    batch_size = embeddings.shape[0]
    
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    similarity_matrix = similarity_matrix / temperature
    
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    
    loss = -mean_log_prob_pos
    loss = loss.mean()
    
    return loss

multiview_tensor = torch.FloatTensor(multiview_embeddings).to(device)
overlap_tensor = torch.FloatTensor(overlap_embeddings).to(device)
combined_tensor = torch.cat([multiview_tensor, overlap_tensor], dim=1)

pin_to_idx_full = {pin: idx for idx, pin in enumerate(all_pins)}

train_full_indices = [pin_to_idx_full[all_pins_labeled[i]] for i in train_indices]
val_full_indices = [pin_to_idx_full[all_pins_labeled[i]] for i in val_indices]
test_full_indices = [pin_to_idx_full[all_pins_labeled[i]] for i in test_indices]

unlabeled_full_indices = [i for i in range(n_total) if all_pins[i] not in pin_to_label]

print(f"Labeled train: {len(train_full_indices)}")
print(f"Labeled val: {len(val_full_indices)}")
print(f"Labeled test: {len(test_full_indices)}")
print(f"Unlabeled: {len(unlabeled_full_indices)}")

compression_params = combined_input_dim * FINAL_DIM
compression_ratio = compression_params / len(train_full_indices)
print(f"Parameters: {compression_params:,}")
print(f"Ratio: 1:{compression_ratio:.1f} {'[SAFE]' if compression_ratio <= MIN_PARAMETER_RATIO else '[RISKY]'}")

compression_model = CompressionNetwork(combined_input_dim, FINAL_DIM).to(device)
optimizer = torch.optim.AdamW(compression_model.parameters(), lr=0.0005, weight_decay=1e-4)

best_val_loss = float('inf')
patience_counter = 0
compression_history = {'train': [], 'val': []}

print("\nStage 1: Training on labeled only (20 epochs)")
for epoch in range(20):
    compression_model.train()
    train_losses = []
    
    np.random.shuffle(train_full_indices)
    n_batches = len(train_full_indices) // BATCH_SIZE
    
    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(train_full_indices))
        batch_idx = train_full_indices[start:end]
        
        batch_x = combined_tensor[batch_idx]
        
        batch_labels = []
        for idx in batch_idx:
            pin = all_pins[idx]
            batch_labels.append(specialty_to_id[pin_to_label[pin]])
        batch_labels = torch.LongTensor(batch_labels).to(device)
        
        compressed = compression_model(batch_x)
        loss = supervised_contrastive_loss(compressed, batch_labels, temperature=0.1)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compression_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
    
    compression_model.eval()
    with torch.no_grad():
        val_batch_losses = []
        n_val_batches = len(val_full_indices) // BATCH_SIZE
        
        for i in range(n_val_batches):
            start = i * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(val_full_indices))
            batch_idx = val_full_indices[start:end]
            
            batch_x = combined_tensor[batch_idx]
            
            batch_labels = []
            for idx in batch_idx:
                pin = all_pins[idx]
                batch_labels.append(specialty_to_id[pin_to_label[pin]])
            batch_labels = torch.LongTensor(batch_labels).to(device)
            
            compressed = compression_model(batch_x)
            loss = supervised_contrastive_loss(compressed, batch_labels, temperature=0.1)
            val_batch_losses.append(loss.item())
    
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_batch_losses)
    
    compression_history['train'].append(train_loss)
    compression_history['val'].append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_compression_state = compression_model.state_dict().copy()
    else:
        patience_counter += 1
    
    is_overfitting, ratio = check_overfitting(train_loss, val_loss)
    status = " [OVERFITTING WARNING]" if is_overfitting else ""
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"Best={best_val_loss:.4f}, Ratio={ratio:.3f}{status}")

compression_model.load_state_dict(best_compression_state)

print("\nGenerating pseudo-labels for unlabeled providers...")
compression_model.eval()

labeled_embeddings = []
labeled_labels = []
with torch.no_grad():
    for idx in train_full_indices:
        x = combined_tensor[idx:idx+1]
        compressed = compression_model(x)
        labeled_embeddings.append(compressed.cpu().numpy())
        pin = all_pins[idx]
        labeled_labels.append(specialty_to_id[pin_to_label[pin]])

labeled_embeddings = np.vstack(labeled_embeddings)
labeled_labels = np.array(labeled_labels)

print(f"Labeled reference set: {labeled_embeddings.shape}")

pseudo_labels = {}
with torch.no_grad():
    for idx in unlabeled_full_indices:
        x = combined_tensor[idx:idx+1]
        compressed = compression_model(x)
        compressed_np = compressed.cpu().numpy()
        
        similarities = np.dot(labeled_embeddings, compressed_np.T).flatten()
        top_k_indices = np.argsort(similarities)[-5:]
        top_k_labels = labeled_labels[top_k_indices]
        
        pseudo_label = np.bincount(top_k_labels).argmax()
        pseudo_labels[idx] = pseudo_label

print(f"Generated pseudo-labels for {len(pseudo_labels)} unlabeled providers")

pseudo_label_counts = defaultdict(int)
for label in pseudo_labels.values():
    pseudo_label_counts[label] += 1

print("\nPseudo-label distribution:")
for spec_id in sorted(pseudo_label_counts.keys()):
    spec_name = id_to_specialty[spec_id]
    count = pseudo_label_counts[spec_id]
    print(f"  {spec_name:25s}: {count:4d}")

print("\nStage 2: Training with pseudo-labels (30 epochs)")
optimizer = torch.optim.AdamW(compression_model.parameters(), lr=0.0003, weight_decay=1e-4)

for epoch in range(30):
    compression_model.train()
    train_losses = []
    
    labeled_batch_indices = train_full_indices.copy()
    unlabeled_batch_indices = list(pseudo_labels.keys())
    
    np.random.shuffle(labeled_batch_indices)
    np.random.shuffle(unlabeled_batch_indices)
    
    n_labeled_batches = len(labeled_batch_indices) // BATCH_SIZE
    n_unlabeled_batches = len(unlabeled_batch_indices) // BATCH_SIZE
    
    for i in range(min(n_labeled_batches, n_unlabeled_batches)):
        labeled_batch = labeled_batch_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        unlabeled_batch = unlabeled_batch_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        
        labeled_x = combined_tensor[labeled_batch]
        labeled_y = torch.LongTensor([
            specialty_to_id[pin_to_label[all_pins[idx]]] 
            for idx in labeled_batch
        ]).to(device)
        
        labeled_compressed = compression_model(labeled_x)
        loss_supervised = supervised_contrastive_loss(labeled_compressed, labeled_y, temperature=0.1)
        
        unlabeled_x = combined_tensor[unlabeled_batch]
        unlabeled_y = torch.LongTensor([
            pseudo_labels[idx] for idx in unlabeled_batch
        ]).to(device)
        
        unlabeled_compressed = compression_model(unlabeled_x)
        loss_pseudo = supervised_contrastive_loss(unlabeled_compressed, unlabeled_y, temperature=0.1)
        
        total_loss = loss_supervised + 0.3 * loss_pseudo
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(compression_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(total_loss.item())
    
    compression_model.eval()
    with torch.no_grad():
        val_batch_losses = []
        n_val_batches = len(val_full_indices) // BATCH_SIZE
        
        for i in range(n_val_batches):
            start = i * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(val_full_indices))
            batch_idx = val_full_indices[start:end]
            
            batch_x = combined_tensor[batch_idx]
            
            batch_labels = []
            for idx in batch_idx:
                pin = all_pins[idx]
                batch_labels.append(specialty_to_id[pin_to_label[pin]])
            batch_labels = torch.LongTensor(batch_labels).to(device)
            
            compressed = compression_model(batch_x)
            loss = supervised_contrastive_loss(compressed, batch_labels, temperature=0.1)
            val_batch_losses.append(loss.item())
    
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_batch_losses)
    
    compression_history['train'].append(train_loss)
    compression_history['val'].append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_compression_state = compression_model.state_dict().copy()
    else:
        patience_counter += 1
    
    is_overfitting, ratio = check_overfitting(train_loss, val_loss)
    status = " [OVERFITTING WARNING]" if is_overfitting else ""
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"Best={best_val_loss:.4f}, Ratio={ratio:.3f}{status}")
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

compression_model.load_state_dict(best_compression_state)

compression_model.eval()
with torch.no_grad():
    test_batch_losses = []
    n_test_batches = len(test_full_indices) // BATCH_SIZE
    
    for i in range(n_test_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(test_full_indices))
        batch_idx = test_full_indices[start:end]
        
        batch_x = combined_tensor[batch_idx]
        
        batch_labels = []
        for idx in batch_idx:
            pin = all_pins[idx]
            batch_labels.append(specialty_to_id[pin_to_label[pin]])
        batch_labels = torch.LongTensor(batch_labels).to(device)
        
        compressed = compression_model(batch_x)
        loss = supervised_contrastive_loss(compressed, batch_labels, temperature=0.1)
        test_batch_losses.append(loss.item())
    
    test_loss = np.mean(test_batch_losses)

final_train = compression_history['train'][-1]
final_val = compression_history['val'][-1]

print(f"\nPhase 3 Complete:")
print(f"  Train loss: {final_train:.4f}")
print(f"  Val loss:   {final_val:.4f}")
print(f"  Test loss:  {test_loss:.4f}")

is_overfitting_val, val_ratio = check_overfitting(final_train, final_val)
is_overfitting_test, test_ratio = check_overfitting(final_train, test_loss)

print(f"\nOverfitting check:")
print(f"  Val/Train:  {val_ratio:.3f} {'[OVERFITTING WARNING]' if is_overfitting_val else '[OK]'}")
print(f"  Test/Train: {test_ratio:.3f} {'[OVERFITTING WARNING]' if is_overfitting_test else '[OK]'}")

torch.save(compression_model.state_dict(), 'phase3_diagnosis_compression_network.pth')

# ============================================================================
# FINAL: GENERATE ALL EMBEDDINGS
# ============================================================================

print("\n" + "="*80)
print("GENERATING FINAL EMBEDDINGS FOR ALL PROVIDERS")
print("="*80)

compression_model.eval()
final_embeddings = []

with torch.no_grad():
    for i in range(0, n_total, BATCH_SIZE_INFERENCE):
        end_idx = min(i + BATCH_SIZE_INFERENCE, n_total)
        batch_x = combined_tensor[i:end_idx]
        
        compressed = compression_model(batch_x)
        compressed_norm = F.normalize(compressed, p=2, dim=1)
        
        final_embeddings.append(compressed_norm.cpu().numpy())

final_embeddings = np.vstack(final_embeddings)
print(f"Final embeddings shape: {final_embeddings.shape}")

np.save('final_diagnosis_embeddings_128d.npy', final_embeddings)

embedding_data = {'PIN': all_pins}
for i in range(FINAL_DIM):
    embedding_data[f'diag_emb_{i}'] = final_embeddings[:, i]

embeddings_df = pd.DataFrame(embedding_data)
embeddings_df.to_parquet('final_diagnosis_embeddings_128d.parquet', index=False)

print("\nSaved: final_diagnosis_embeddings_128d.npy")
print("Saved: final_diagnosis_embeddings_128d.parquet")

final_metadata = {
    'pipeline': 'unified_diagnosis_with_overlap_semisupervised',
    'phase1_latent_dim': PHASE1_LATENT_DIM,
    'phase2_multiview_dim': multiview_dim,
    'overlap_embedding_dim': OVERLAP_EMBEDDING_DIM,
    'combined_input_dim': combined_input_dim,
    'final_dim': FINAL_DIM,
    'total_providers': n_total,
    'labeled_providers': n_labeled,
    'unlabeled_providers': len(unlabeled_full_indices),
    'num_specialties': num_specialties,
    'specialty_to_id': specialty_to_id,
    'specialty_code_indices': all_specialty_codes,
    'original_diagnosis_dim': diag_matrix.shape[1],
    'phase1_val_train_ratio': phase1_val_ratio,
    'phase1_test_train_ratio': phase1_test_ratio,
    'phase3_val_train_ratio': val_ratio,
    'phase3_test_train_ratio': test_ratio,
    'semi_supervised': True,
    'pseudo_label_count': len(pseudo_labels)
}

with open('final_diagnosis_pipeline_metadata.pkl', 'wb') as f:
    pickle.dump(final_metadata, f)

print("Saved: final_diagnosis_pipeline_metadata.pkl")

print("\n" + "="*80)
print("PIPELINE COMPLETE")
print("="*80)
print(f"\nFinal outputs:")
print(f"  1. final_diagnosis_embeddings_128d.npy - Embeddings for all {n_total} providers")
print(f"  2. final_diagnosis_embeddings_128d.parquet - Embeddings with PIN column")
print(f"  3. final_diagnosis_pipeline_metadata.pkl - Complete metadata")
print(f"\nPipeline stages:")
print(f"  Phase 1: {input_dim} → {PHASE1_LATENT_DIM} (specialty-conditioned)")
print(f"  Phase 2: {PHASE1_LATENT_DIM} × {num_specialties} = {multiview_dim} (multi-view)")
print(f"  Phase 2.5: {input_dim} → {OVERLAP_EMBEDDING_DIM} (overlap autoencoder)")
print(f"  Phase 3: {combined_input_dim} → {FINAL_DIM} (semi-supervised compression)")
print(f"\nSemi-supervised training:")
print(f"  Labeled: {n_labeled} providers")
print(f"  Pseudo-labeled: {len(pseudo_labels)} providers")
print(f"  Total trained on: {n_labeled + len(pseudo_labels)} providers")
