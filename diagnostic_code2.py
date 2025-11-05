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
