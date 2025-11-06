print("\n" + "="*80)
print("CHECKING FOR PROVIDERS WITH EMPTY/INVALID CODE EMBEDDINGS")
print("="*80)

empty_providers = []
small_providers = []
valid_providers = []

for provider_idx, data in provider_code_data.items():
    n_codes = data['code_embs'].shape[0]
    
    if n_codes == 0:
        empty_providers.append(provider_idx)
    elif n_codes == 1:
        small_providers.append(provider_idx)
    else:
        valid_providers.append(provider_idx)

print(f"\nResults:")
print(f"  Empty (0 codes):  {len(empty_providers)} providers")
print(f"  Small (1 code):   {len(small_providers)} providers")
print(f"  Valid (2+ codes): {len(valid_providers)} providers")
print(f"  Total:            {len(provider_code_data)} providers")

if empty_providers:
    print(f"\n⚠️  WARNING: Found {len(empty_providers)} providers with NO code embeddings!")
    print(f"Sample empty provider indices: {empty_providers[:10]}")
    
    print("\nChecking why they're empty...")
    for idx in empty_providers[:3]:
        pin = all_pins[idx]
        provider_codes = proc_matrix_filtered[idx].nonzero()[1]
        print(f"\n  Provider {idx} (PIN={pin}):")
        print(f"    Raw procedure codes: {len(provider_codes)}")
        print(f"    Codes that exist in code_embeddings: 0")
        if len(provider_codes) > 0:
            sample_codes = provider_codes[:5]
            for code_idx in sample_codes:
                code_id = all_specialty_codes[code_idx]
                in_graph = code_id in code_embeddings
                print(f"      Code {code_id}: in graph={in_graph}")

if small_providers:
    print(f"\n⚠️  INFO: {len(small_providers)} providers have only 1 code")
    print(f"    (Might cause issues with attention mechanism)")

print("\n" + "="*80)
print("RECOMMENDATION:")
if empty_providers:
    print("  Remove empty providers from provider_code_data before training")
    print("  Add this after creating provider_code_data:")
    print("  provider_code_data = {k: v for k, v in provider_code_data.items() if v['code_embs'].shape[0] > 0}")
if small_providers:
    print("  Consider filtering providers with only 1 code (optional)")
print("="*80)



def attention_head(self, provider_emb, code_embs, W, a):
    provider_h = W(provider_emb)
    code_h = W(code_embs)
    
    n_codes = code_embs.shape[0]
    provider_repeated = provider_h.repeat(n_codes, 1)
    
    concat = torch.cat([provider_repeated, code_h], dim=1)
    e = self.leaky_relu(concat @ a).squeeze(-1)  # CHANGE: squeeze only last dim
    
    # CHANGE: Handle single code case
    if e.dim() == 0:
        e = e.unsqueeze(0)
    
    alpha = F.softmax(e, dim=0)
    alpha = self.dropout(alpha)
    
    aggregated = (alpha.unsqueeze(1) * code_h).sum(dim=0)
    return aggregated




# OLD (fails):
final_embeddings[provider_idx] = provider_init_embeddings[provider_idx, :512]
# Tries to slice 128D array with [:512] → error

# NEW (works):
final_embeddings[provider_idx, :CODE_EMBEDDING_DIM] = provider_init_embeddings[provider_idx]
# Copies 128D into first 128 positions of 512D array
# Remaining 384 positions stay as 0


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
                code_id = all_specialty_codes[code_idx]
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
print("STEP 5: SAVE EMBEDDINGS TO CSV")
print("="*80)

specialty_column = [pin_to_label.get(pin, 'UNLABELED') for pin in all_pins]

embedding_dict = {'PIN': all_pins}
for i in range(final_embeddings.shape[1]):
    embedding_dict[f'emb_{i}'] = final_embeddings[:, i]
embedding_dict['specialty'] = specialty_column

embedding_df = pd.DataFrame(embedding_dict)

output_csv = 'me2vec_provider_embeddings.csv'
embedding_df.to_csv(output_csv, index=False, float_format='%.6f')
print(f"\nSaved embeddings to: {output_csv}")
print(f"Shape: {embedding_df.shape}")
print(f"Columns: PIN, emb_0 to emb_{final_embeddings.shape[1]-1}, specialty")

print("\nFirst 5 rows preview:")
print(embedding_df.head(5)[['PIN', 'emb_0', 'emb_1', 'emb_2', 'specialty']])

np.save('me2vec_provider_embeddings.npy', final_embeddings)
print(f"\nAlso saved as numpy array: me2vec_provider_embeddings.npy")

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
    'q': q
}

with open('me2vec_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("Saved metadata: me2vec_metadata.pkl")

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print("\nOutputs:")
print("  1. me2vec_provider_embeddings.csv - CSV with PIN and embeddings")
print("  2. me2vec_provider_embeddings.npy - NumPy array of embeddings")
print("  3. me2vec_metadata.pkl - Pipeline metadata")
print("\nEmbedding pipeline:")
print(f"  Step 1: {n_codes} procedure codes -> {CODE_EMBEDDING_DIM}D (biased random walk)")
print(f"  Step 2: Initialize {len(all_pins)} providers from weighted code embeddings")
print(f"  Step 3: Train GAT with {num_specialties} specialties")
print(f"  Step 4: Final embeddings: {final_embeddings.shape[1]}D per provider")
