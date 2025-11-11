import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import load_npz
from collections import defaultdict
import networkx as nx
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CODE_EMBEDDING_DIM = 128
PROVIDER_EMBEDDING_DIM = 128
NUM_HEADS = 4
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
RARITY_THRESHOLD = 0.70

print("\n" + "="*80)
print("ME2VEC WITH RARE CODES ONLY (RARITY SCORE >= 0.70)")
print("="*80)

print("\nLoading data...")
proc_matrix = load_npz('procedure_vectors.npz')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

print(f"\nLoading rare codes from Part 1...")
rare_codes_df = pd.read_parquet('rare_codes.parquet')
rare_codes_df = rare_codes_df[rare_codes_df['rarity_score'] >= RARITY_THRESHOLD]

print(f"\nRare codes with rarity_score >= {RARITY_THRESHOLD}: {len(rare_codes_df)}")
print(f"Top 10 rare codes:")
print(rare_codes_df.head(10)[['code', 'total_claims', 'num_providers', 'rarity_score']])

print(f"\nLoading procedure code mapping...")
procedure_df = pd.read_parquet('procedure_df.parquet')
unique_codes = sorted(procedure_df['code'].unique())
code_to_idx = {code: idx for idx, code in enumerate(unique_codes)}

rare_code_indices = [code_to_idx[code] for code in rare_codes_df['code'] if code in code_to_idx]
rare_codes_list = [code for code in rare_codes_df['code'] if code in code_to_idx]

print(f"\nMatched rare codes to matrix indices: {len(rare_code_indices)}")

print(f"\nTotal providers: {len(all_pins)}")
print(f"Labeled providers: {len([p for p in all_pins if p in pin_to_label])}")
print(f"Unlabeled providers: {len([p for p in all_pins if p not in pin_to_label])}")
print(f"Total procedure codes in matrix: {proc_matrix.shape[1]}")
print(f"Rare codes selected: {len(rare_code_indices)}")

proc_matrix_filtered = proc_matrix[:, rare_code_indices]

print("\n" + "="*80)
print("STEP 1: PROCEDURE CODE EMBEDDING (Node2Vec-style)")
print("="*80)

print("\nBuilding procedure co-occurrence graph (vectorized)...")
n_providers, n_codes = proc_matrix_filtered.shape

proc_matrix_binary = (proc_matrix_filtered > 0).astype(float)
cooccurrence_matrix = proc_matrix_binary.T @ proc_matrix_binary
del proc_matrix_binary

cooccurrence_matrix = cooccurrence_matrix.tocoo()

print(f"Total co-occurrence edges: {cooccurrence_matrix.nnz}")

G = nx.Graph()
for i, j, weight in zip(cooccurrence_matrix.row, cooccurrence_matrix.col, cooccurrence_matrix.data):
    if i < j and weight > 0:
        code1 = rare_code_indices[i]
        code2 = rare_code_indices[j]
        G.add_edge(code1, code2, weight=weight)

print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Graph edges: {G.number_of_edges()}")

print("\nGenerating biased random walks (parallel)...")
walk_length = 80
num_walks = 10
p = 4.0
q = 1.0

neighbor_cache = {node: list(G.neighbors(node)) for node in G.nodes()}
neighbor_set_cache = {node: set(neighbors) for node, neighbors in neighbor_cache.items()}

def biased_random_walk_optimized(args):
    start_node, walk_length, p, q, neighbor_cache, neighbor_set_cache = args
    walk = [start_node]
    
    neighbors = neighbor_cache[start_node]
    if len(neighbors) == 0:
        return [str(n) for n in walk]
    
    walk.append(np.random.choice(neighbors))
    
    for _ in range(walk_length - 2):
        cur = walk[-1]
        prev = walk[-2]
        neighbors = neighbor_cache[cur]
        
        if len(neighbors) == 0:
            break
        
        prev_neighbors = neighbor_set_cache[prev]
        
        probs = np.array([
            1.0/p if n == prev else (1.0 if n in prev_neighbors else 1.0/q)
            for n in neighbors
        ], dtype=np.float32)
        
        probs = probs / probs.sum()
        next_node = np.random.choice(neighbors, p=probs)
        walk.append(next_node)
    
    return [str(n) for n in walk]

from multiprocessing import Pool, cpu_count
num_workers = min(cpu_count(), 8)

nodes = list(G.nodes())
walk_tasks = []
for walk_iter in range(num_walks):
    shuffled_nodes = nodes.copy()
    np.random.shuffle(shuffled_nodes)
    for node in shuffled_nodes:
        walk_tasks.append((node, walk_length, p, q, neighbor_cache, neighbor_set_cache))

print(f"Generating {len(walk_tasks)} walks using {num_workers} workers...")

with Pool(num_workers) as pool:
    walks = pool.map(biased_random_walk_optimized, walk_tasks, chunksize=100)

del neighbor_set_cache

print(f"Generated {len(walks)} walks")

print("\nTraining Word2Vec on walks...")
w2v_model = Word2Vec(walks, vector_size=CODE_EMBEDDING_DIM, window=10, 
                     min_count=1, sg=1, workers=8, epochs=10, negative=5, ns_exponent=0.75)

code_embeddings = {}
for node in G.nodes():
    code_embeddings[node] = w2v_model.wv[str(node)]

print(f"Trained embeddings for {len(code_embeddings)} procedure codes")

del walks, G, neighbor_cache, cooccurrence_matrix
import gc
gc.collect()

print("\n" + "="*80)
print("STEP 2: INITIALIZE PROVIDER EMBEDDINGS")
print("="*80)

print("\nComputing weighted average of code embeddings for each provider...")
provider_init_embeddings = np.zeros((len(all_pins), CODE_EMBEDDING_DIM), dtype=np.float32)

for provider_idx in range(len(all_pins)):
    provider_codes = proc_matrix_filtered[provider_idx].nonzero()[1]
    provider_claims = proc_matrix_filtered[provider_idx].data
    
    if len(provider_codes) == 0:
        continue
    
    total_weight = 0
    for code_idx, claim_count in zip(provider_codes, provider_claims):
        code_id = rare_code_indices[code_idx]
        if code_id in code_embeddings:
            provider_init_embeddings[provider_idx] += code_embeddings[code_id] * claim_count
            total_weight += claim_count
    
    if total_weight > 0:
        provider_init_embeddings[provider_idx] /= total_weight
print(f"Initialized {len(provider_init_embeddings)} provider embeddings")

print("\n" + "="*80)
print("STEP 3: TRAIN GAT FOR SPECIALTY PREDICTION")
print("="*80)

class ProviderGAT(nn.Module):
    def __init__(self, code_dim, provider_dim, num_specialties, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.code_dim = code_dim
        self.provider_dim = provider_dim
        self.head_dim = provider_dim
        
        self.W_heads = nn.ModuleList([
            nn.Linear(code_dim, self.head_dim, bias=False) 
            for _ in range(num_heads)
        ])
        
        self.a_heads = nn.ParameterList([
            nn.Parameter(torch.randn(2 * self.head_dim, 1))
            for _ in range(num_heads)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        self.specialty_classifier = nn.Sequential(
            nn.Linear(provider_dim * num_heads, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_specialties)
        )
    
    def forward(self, provider_emb, code_embs):
        batch_size = provider_emb.size(0)
        n_codes = code_embs.size(0)
        
        head_outputs = []
        
        for head_idx in range(self.num_heads):
            W = self.W_heads[head_idx]
            a = self.a_heads[head_idx]
            
            provider_h = W(provider_emb)
            code_h = W(code_embs)
            
            provider_repeated = provider_h.repeat(n_codes, 1)
            concat = torch.cat([provider_repeated, code_h], dim=1)
            
            e = self.leakyrelu(concat @ a).squeeze(-1)
            attention = F.softmax(e, dim=0)
            attention = self.dropout(attention)
            
            aggregated = (attention.unsqueeze(1) * code_h).sum(dim=0, keepdim=True)
            head_outputs.append(aggregated)
        
        combined = torch.cat(head_outputs, dim=1)
        specialty_logits = self.specialty_classifier(combined).squeeze(0)
        
        return specialty_logits, combined.squeeze(0)

print("\nPreparing labeled data for training...")
unique_specialties = sorted(set(pin_to_label.values()))
specialty_to_id = {spec: idx for idx, spec in enumerate(unique_specialties)}
num_specialties = len(unique_specialties)

print(f"Number of unique specialties: {num_specialties}")

provider_code_data = {}

for provider_idx, pin in enumerate(all_pins):
    if pin not in pin_to_label:
        continue
    
    provider_codes = proc_matrix_filtered[provider_idx].nonzero()[1]
    
    code_emb_list = []
    for code_idx in provider_codes:
        code_id = rare_code_indices[code_idx]
        if code_id in code_embeddings:
            code_emb_list.append(code_embeddings[code_id])
    
    if len(code_emb_list) > 0:
        provider_code_data[provider_idx] = {
            'code_embs': np.array(code_emb_list, dtype=np.float32),
            'specialty_id': specialty_to_id[pin_to_label[pin]]
        }

valid_labeled_indices = list(provider_code_data.keys())
print(f"Valid providers for training: {len(valid_labeled_indices)}")

model = ProviderGAT(CODE_EMBEDDING_DIM, PROVIDER_EMBEDDING_DIM, 
                    num_specialties, NUM_HEADS, dropout=0.3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print("\nTraining GAT model...")

for epoch in range(EPOCHS):
    model.train()
    
    np.random.shuffle(valid_labeled_indices)
    batch_indices = [valid_labeled_indices[i:i+BATCH_SIZE] 
                     for i in range(0, len(valid_labeled_indices), BATCH_SIZE)]
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_idx_list in batch_indices:
        batch_loss = 0
        batch_correct = 0
        batch_total = 0
        
        for provider_idx in batch_idx_list:
            data = provider_code_data[provider_idx]
            
            provider_emb = torch.FloatTensor(provider_init_embeddings[provider_idx]).unsqueeze(0).to(device)
            code_embs = torch.FloatTensor(data['code_embs']).to(device)
            specialty_label = torch.LongTensor([data['specialty_id']]).to(device)
            
            specialty_logits, _ = model(provider_emb, code_embs)
            loss = criterion(specialty_logits.unsqueeze(0), specialty_label)
            
            batch_loss += loss
            pred = specialty_logits.argmax()
            batch_correct += (pred == specialty_label[0]).item()
            batch_total += 1
        
        if batch_total > 0:
            optimizer.zero_grad()
            (batch_loss / batch_total).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            correct += batch_correct
            total += batch_total
    
    if (epoch + 1) % 10 == 0:
        acc = correct / total if total > 0 else 0
        avg_loss = epoch_loss / total if total > 0 else 0
        print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Accuracy={acc:.3f}")

del provider_code_data
gc.collect()

print("\n" + "="*80)
print("STEP 4: GENERATE FINAL PROVIDER EMBEDDINGS")
print("="*80)

model.eval()
final_embeddings = np.zeros((len(all_pins), PROVIDER_EMBEDDING_DIM * NUM_HEADS), dtype=np.float32)

print("\nGenerating embeddings for all providers (batched)...")
INFERENCE_BATCH_SIZE = 128

with torch.no_grad():
    for batch_start in range(0, len(all_pins), INFERENCE_BATCH_SIZE):
        batch_end = min(batch_start + INFERENCE_BATCH_SIZE, len(all_pins))
        
        for provider_idx in range(batch_start, batch_end):
            provider_codes = proc_matrix_filtered[provider_idx].nonzero()[1]
            
            if len(provider_codes) == 0:
                final_embeddings[provider_idx] = np.tile(provider_init_embeddings[provider_idx], NUM_HEADS)
                continue
            
            code_emb_list = []
            for code_idx in provider_codes:
                code_id = rare_code_indices[code_idx]
                if code_id in code_embeddings:
                    code_emb_list.append(code_embeddings[code_id])
            
            if len(code_emb_list) == 0:
                final_embeddings[provider_idx] = np.tile(provider_init_embeddings[provider_idx], NUM_HEADS)
                continue
            
            provider_emb = torch.FloatTensor(provider_init_embeddings[provider_idx]).unsqueeze(0).to(device)
            code_embs = torch.FloatTensor(np.array(code_emb_list, dtype=np.float32)).to(device)
            
            _, updated_emb = model(provider_emb, code_embs)
            final_embeddings[provider_idx] = updated_emb.cpu().numpy()
        
        if (batch_end) % 1000 == 0 or batch_end == len(all_pins):
            print(f"  Processed {batch_end}/{len(all_pins)} providers")
print(f"Generated embeddings shape: {final_embeddings.shape}")

print("\n" + "="*80)
print("STEP 5: SAVE EMBEDDINGS")
print("="*80)

specialty_column = [pin_to_label.get(pin, 'UNLABELED') for pin in all_pins]

embedding_dict = {'PIN': all_pins}
for i in range(final_embeddings.shape[1]):
    embedding_dict[f'emb_{i}'] = final_embeddings[:, i]
embedding_dict['specialty'] = specialty_column

embedding_df = pd.DataFrame(embedding_dict)

output_csv = 'rare_codes_provider_embeddings.csv'
embedding_df.to_csv(output_csv, index=False, float_format='%.6f')
print(f"\nSaved embeddings to: {output_csv}")
print(f"Shape: {embedding_df.shape}")
print(f"Columns: PIN, emb_0 to emb_{final_embeddings.shape[1]-1}, specialty")

print("\nFirst 5 rows preview:")
print(embedding_df.head(5)[['PIN', 'emb_0', 'emb_1', 'emb_2', 'specialty']])

np.save('rare_codes_provider_embeddings.npy', final_embeddings)
print(f"\nAlso saved as numpy array: rare_codes_provider_embeddings.npy")

metadata = {
    'code_embedding_dim': CODE_EMBEDDING_DIM,
    'provider_embedding_dim': PROVIDER_EMBEDDING_DIM,
    'num_heads': NUM_HEADS,
    'num_providers': len(all_pins),
    'num_labeled': len(pin_to_label),
    'num_specialties': num_specialties,
    'specialty_to_id': specialty_to_id,
    'walk_length': walk_length,
    'num_walks': num_walks,
    'p': p,
    'q': q,
    'rarity_threshold': RARITY_THRESHOLD,
    'num_rare_codes': len(rare_code_indices)
}

with open('rare_codes_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("Saved metadata: rare_codes_metadata.pkl")

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print("\nOutputs:")
print("  1. rare_codes_provider_embeddings.csv - CSV with PIN and embeddings")
print("  2. rare_codes_provider_embeddings.npy - NumPy array of embeddings")
print("  3. rare_codes_metadata.pkl - Pipeline metadata")
print(f"\nEmbedding pipeline (RARE CODES ONLY):")
print(f"  Rarity threshold: >= {RARITY_THRESHOLD}")
print(f"  Rare codes used: {len(rare_code_indices)}")
print(f"  Step 1: Procedure code embeddings -> {CODE_EMBEDDING_DIM}D (biased random walk)")
print(f"  Step 2: Initialize {len(all_pins)} providers from weighted code embeddings")
print(f"  Step 3: Train GAT with {num_specialties} specialties")
print(f"  Step 4: Final embeddings: {final_embeddings.shape[1]}D per provider")
