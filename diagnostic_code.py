print("\n" + "="*80)
print("DIAGNOSTIC: Shape debugging")
print("="*80)

print(f"Model created. Checking architecture...")
print(f"CODE_EMBEDDING_DIM: {CODE_EMBEDDING_DIM}")
print(f"PROVIDER_EMBEDDING_DIM: {PROVIDER_EMBEDDING_DIM}")
print(f"NUM_HEADS: {NUM_HEADS}")
print(f"num_specialties: {num_specialties}")

print(f"\nProvider init embeddings shape: {provider_init_embeddings.shape}")
print(f"Number of valid labeled indices: {len(valid_labeled_indices)}")

if len(valid_labeled_indices) > 0:
    first_idx = valid_labeled_indices[0]
    first_data = provider_code_data[first_idx]
    
    print(f"\nFirst sample (provider_idx={first_idx}):")
    print(f"  provider_init_embeddings[{first_idx}].shape = {provider_init_embeddings[first_idx].shape}")
    print(f"  code_embs shape = {first_data['code_embs'].shape}")
    print(f"  specialty_id = {first_data['specialty_id']}")
    
    test_provider_emb = torch.FloatTensor(provider_init_embeddings[first_idx]).unsqueeze(0).to(device)
    test_code_embs = torch.FloatTensor(first_data['code_embs']).to(device)
    
    print(f"\nTensor shapes after conversion:")
    print(f"  test_provider_emb.shape = {test_provider_emb.shape}")
    print(f"  test_code_embs.shape = {test_code_embs.shape}")
    
    print(f"\nTesting model forward pass...")
    try:
        with torch.no_grad():
            test_logits, test_emb = model(test_provider_emb, test_code_embs)
        print(f"✓ Forward pass SUCCESS")
        print(f"  test_logits.shape = {test_logits.shape}")
        print(f"  test_emb.shape = {test_emb.shape}")
    except Exception as e:
        print(f"✗ Forward pass FAILED: {e}")
        print(f"\nDebugging attention_head...")
        
        W = model.W_heads[0]
        a = model.a_heads[0]
        
        print(f"  W input_dim={CODE_EMBEDDING_DIM}, output_dim={PROVIDER_EMBEDDING_DIM}")
        print(f"  a.shape = {a.shape}")
        
        provider_h = W(test_provider_emb)
        code_h = W(test_code_embs)
        print(f"  provider_h.shape = {provider_h.shape}")
        print(f"  code_h.shape = {code_h.shape}")
        
        n_codes = test_code_embs.shape[0]
        print(f"  n_codes = {n_codes}")
        
        provider_repeated = provider_h.repeat(n_codes, 1)
        print(f"  provider_repeated.shape = {provider_repeated.shape}")
        
        concat = torch.cat([provider_repeated, code_h], dim=1)
        print(f"  concat.shape = {concat.shape}")
        print(f"  Expected shape for (concat @ a): [{n_codes}, 1]")
        
        try:
            e = concat @ a
            print(f"  e.shape after concat @ a = {e.shape}")
        except Exception as e2:
            print(f"  ✗ ERROR at concat @ a: {e2}")

print("="*80)
