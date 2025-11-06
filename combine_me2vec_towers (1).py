import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, normalize

print("\n" + "="*80)
print("COMBINING ME2VEC EMBEDDINGS WITH LINEAR TOWERS")
print("="*80)

print("\nLoading data...")

proc_emb_df = pd.read_parquet('me2vec_provider_embeddings.parquet')
diag_emb_df = pd.read_parquet('me2vec_provider_embeddings_diagnosis.parquet')

proc_emb_cols = [col for col in proc_emb_df.columns if col.startswith('emb_')]
diag_emb_cols = [col for col in diag_emb_df.columns if col.startswith('emb_')]

proc_embeddings = proc_emb_df[proc_emb_cols].values
diag_embeddings = diag_emb_df[diag_emb_cols].values

proc_pins = proc_emb_df['PIN'].tolist()
diag_pins = diag_emb_df['PIN'].tolist()

print(f"\nLoaded embeddings:")
print(f"  Procedure embeddings: {proc_embeddings.shape}")
print(f"  Diagnosis embeddings: {diag_embeddings.shape}")
print(f"  Procedure PINs: {len(proc_pins)}")
print(f"  Diagnosis PINs: {len(diag_pins)}")

print("\n" + "="*80)
print("ALIGNING PINS ACROSS MODALITIES")
print("="*80)

proc_pin_set = set(proc_pins)
diag_pin_set = set(diag_pins)

all_pins_union = sorted(proc_pin_set | diag_pin_set)
pins_in_both = proc_pin_set & diag_pin_set
pins_only_proc = proc_pin_set - diag_pin_set
pins_only_diag = diag_pin_set - proc_pin_set

print(f"\nPIN statistics:")
print(f"  Total unique PINs: {len(all_pins_union)}")
print(f"  PINs in both: {len(pins_in_both)}")
print(f"  PINs only in procedures: {len(pins_only_proc)}")
print(f"  PINs only in diagnoses: {len(pins_only_diag)}")

proc_pin_to_idx = {pin: idx for idx, pin in enumerate(proc_pins)}
diag_pin_to_idx = {pin: idx for idx, pin in enumerate(diag_pins)}

aligned_proc_embeddings = np.zeros((len(all_pins_union), proc_embeddings.shape[1]), dtype=np.float32)
aligned_diag_embeddings = np.zeros((len(all_pins_union), diag_embeddings.shape[1]), dtype=np.float32)

for new_idx, pin in enumerate(all_pins_union):
    if pin in proc_pin_to_idx:
        old_idx = proc_pin_to_idx[pin]
        aligned_proc_embeddings[new_idx] = proc_embeddings[old_idx]
    
    if pin in diag_pin_to_idx:
        old_idx = diag_pin_to_idx[pin]
        aligned_diag_embeddings[new_idx] = diag_embeddings[old_idx]

print(f"\nAligned embeddings:")
print(f"  Procedure: {aligned_proc_embeddings.shape}")
print(f"  Diagnosis: {aligned_diag_embeddings.shape}")

print("\n" + "="*80)
print("L2 NORMALIZING NEURAL EMBEDDINGS")
print("="*80)

proc_embeddings_norm = normalize(aligned_proc_embeddings, norm='l2', axis=1)
diag_embeddings_norm = normalize(aligned_diag_embeddings, norm='l2', axis=1)

print(f"\nAfter L2 normalization:")
print(f"  Procedure: mean={proc_embeddings_norm.mean():.4f}, std={proc_embeddings_norm.std():.4f}")
print(f"  Diagnosis: mean={diag_embeddings_norm.mean():.4f}, std={diag_embeddings_norm.std():.4f}")

proc_norms = np.linalg.norm(proc_embeddings_norm, axis=1)
diag_norms = np.linalg.norm(diag_embeddings_norm, axis=1)
print(f"  Procedure L2 norms: min={proc_norms.min():.4f}, max={proc_norms.max():.4f}")
print(f"  Diagnosis L2 norms: min={diag_norms.min():.4f}, max={diag_norms.max():.4f}")

print("\n" + "="*80)
print("LOADING LINEAR TOWERS")
print("="*80)

member_matrix = np.load('member_vectors.npy')

with open('vectors_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

all_pins_linear = np.load('all_pins.npy', allow_pickle=True).tolist()

print(f"\nLoaded linear towers:")
print(f"  Member matrix: {member_matrix.shape}")
print(f"  Linear PINs: {len(all_pins_linear)}")
print(f"  Metadata n_member: {metadata['n_member']}")

print("\n" + "="*80)
print("ALIGNING LINEAR TOWERS WITH ME2VEC PINS")
print("="*80)

linear_pin_set = set(all_pins_linear)
final_pins = sorted(set(all_pins_union) & linear_pin_set)

print(f"\nFinal PIN alignment:")
print(f"  ME2Vec PINs: {len(all_pins_union)}")
print(f"  Linear PINs: {len(all_pins_linear)}")
print(f"  Intersection (final): {len(final_pins)}")

me2vec_pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins_union)}
linear_pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins_linear)}

final_proc_embeddings = np.array([proc_embeddings_norm[me2vec_pin_to_idx[pin]] for pin in final_pins])
final_diag_embeddings = np.array([diag_embeddings_norm[me2vec_pin_to_idx[pin]] for pin in final_pins])
final_member_matrix = np.array([member_matrix[linear_pin_to_idx[pin]] for pin in final_pins])

print(f"\nFinal aligned data:")
print(f"  Procedure embeddings: {final_proc_embeddings.shape}")
print(f"  Diagnosis embeddings: {final_diag_embeddings.shape}")
print(f"  Member matrix: {final_member_matrix.shape}")

print("\n" + "="*80)
print("SPLITTING MEMBER MATRIX INTO LINEAR TOWERS")
print("="*80)

demo_start, demo_end = 0, 5
place_start, place_end = 5, 9
cost_start, cost_end = 9, 20
pin_start, pin_end = 20, 22

demographics = final_member_matrix[:, demo_start:demo_end]
place = final_member_matrix[:, place_start:place_end]
cost = final_member_matrix[:, cost_start:cost_end]
pin_summary = final_member_matrix[:, pin_start:pin_end]

print(f"\nSplit member matrix:")
print(f"  Demographics: {demographics.shape} (columns {demo_start}:{demo_end})")
print(f"  Place: {place.shape} (columns {place_start}:{place_end})")
print(f"  Cost: {cost.shape} (columns {cost_start}:{cost_end})")
print(f"  PIN summary: {pin_summary.shape} (columns {pin_start}:{pin_end})")

total_linear = demographics.shape[1] + place.shape[1] + cost.shape[1] + pin_summary.shape[1]
print(f"  Total linear dims: {total_linear}")

print("\n" + "="*80)
print("NORMALIZING LINEAR TOWERS")
print("="*80)

scaler_demo = StandardScaler()
scaler_place = StandardScaler()
scaler_cost = StandardScaler()
scaler_pin = StandardScaler()

demographics_norm = scaler_demo.fit_transform(demographics)
place_norm = scaler_place.fit_transform(place)
cost_norm = scaler_cost.fit_transform(cost)
pin_norm = scaler_pin.fit_transform(pin_summary)

print(f"\nNormalized linear towers:")
print(f"  Demographics: mean={demographics_norm.mean():.4f}, std={demographics_norm.std():.4f}")
print(f"  Place: mean={place_norm.mean():.4f}, std={place_norm.std():.4f}")
print(f"  Cost: mean={cost_norm.mean():.4f}, std={cost_norm.std():.4f}")
print(f"  PIN: mean={pin_norm.mean():.4f}, std={pin_norm.std():.4f}")

with open('me2vec_linear_towers_scalers.pkl', 'wb') as f:
    pickle.dump({
        'scaler_demo': scaler_demo,
        'scaler_place': scaler_place,
        'scaler_cost': scaler_cost,
        'scaler_pin': scaler_pin
    }, f)
print("\n✓ Saved scalers: me2vec_linear_towers_scalers.pkl")

print("\n" + "="*80)
print("COMBINING ALL 6 TOWERS")
print("="*80)

all_embeddings = np.hstack([
    final_proc_embeddings,
    final_diag_embeddings,
    demographics_norm,
    place_norm,
    cost_norm,
    pin_norm
])

print(f"\nCombined embeddings shape: {all_embeddings.shape}")
print(f"  Tower 1 (Procedures):   {final_proc_embeddings.shape[1]} dims")
print(f"  Tower 2 (Diagnoses):    {final_diag_embeddings.shape[1]} dims")
print(f"  Tower 3 (Demographics): {demographics_norm.shape[1]} dims")
print(f"  Tower 4 (Place):        {place_norm.shape[1]} dims")
print(f"  Tower 5 (Cost):         {cost_norm.shape[1]} dims")
print(f"  Tower 6 (PIN):          {pin_norm.shape[1]} dims")
print(f"  Total:                  {all_embeddings.shape[1]} dims")

print("\n" + "="*80)
print("CREATING FINAL DATAFRAME")
print("="*80)

embedding_data = {'PIN': final_pins}

for i in range(final_proc_embeddings.shape[1]):
    embedding_data[f'tower1_proc_emb_{i}'] = final_proc_embeddings[:, i]

for i in range(final_diag_embeddings.shape[1]):
    embedding_data[f'tower2_diag_emb_{i}'] = final_diag_embeddings[:, i]

for i in range(demographics_norm.shape[1]):
    embedding_data[f'tower3_demo_emb_{i}'] = demographics_norm[:, i]

for i in range(place_norm.shape[1]):
    embedding_data[f'tower4_plc_emb_{i}'] = place_norm[:, i]

for i in range(cost_norm.shape[1]):
    embedding_data[f'tower5_cost_emb_{i}'] = cost_norm[:, i]

for i in range(pin_norm.shape[1]):
    embedding_data[f'tower6_pin_emb_{i}'] = pin_norm[:, i]

final_df = pd.DataFrame(embedding_data)

print(f"\nFinal DataFrame shape: {final_df.shape}")
print(f"  Columns: PIN + {final_df.shape[1]-1} embedding dimensions")

output_parquet = 'final_me2vec_all_towers_1046d.parquet'
output_npy = 'final_me2vec_all_towers_1046d.npy'

final_df.to_parquet(output_parquet, index=False)
np.save(output_npy, all_embeddings)

print(f"\n✓ Saved: {output_parquet}")
print(f"✓ Saved: {output_npy}")

print("\n" + "="*80)
print("SAVING METADATA")
print("="*80)

final_metadata = {
    'total_dims': all_embeddings.shape[1],
    'n_providers': all_embeddings.shape[0],
    'tower_dims': {
        'tower1_procedures': final_proc_embeddings.shape[1],
        'tower2_diagnoses': final_diag_embeddings.shape[1],
        'tower3_demographics': demographics_norm.shape[1],
        'tower4_place': place_norm.shape[1],
        'tower5_cost': cost_norm.shape[1],
        'tower6_pin': pin_norm.shape[1]
    },
    'tower_ranges': {
        'tower1_procedures': (0, final_proc_embeddings.shape[1]),
        'tower2_diagnoses': (final_proc_embeddings.shape[1], 
                            final_proc_embeddings.shape[1] + final_diag_embeddings.shape[1]),
        'tower3_demographics': (final_proc_embeddings.shape[1] + final_diag_embeddings.shape[1],
                               final_proc_embeddings.shape[1] + final_diag_embeddings.shape[1] + demographics_norm.shape[1]),
        'tower4_place': (final_proc_embeddings.shape[1] + final_diag_embeddings.shape[1] + demographics_norm.shape[1],
                        final_proc_embeddings.shape[1] + final_diag_embeddings.shape[1] + demographics_norm.shape[1] + place_norm.shape[1]),
        'tower5_cost': (final_proc_embeddings.shape[1] + final_diag_embeddings.shape[1] + demographics_norm.shape[1] + place_norm.shape[1],
                       final_proc_embeddings.shape[1] + final_diag_embeddings.shape[1] + demographics_norm.shape[1] + place_norm.shape[1] + cost_norm.shape[1]),
        'tower6_pin': (final_proc_embeddings.shape[1] + final_diag_embeddings.shape[1] + demographics_norm.shape[1] + place_norm.shape[1] + cost_norm.shape[1],
                      all_embeddings.shape[1])
    },
    'normalization': {
        'tower1_procedures': 'L2',
        'tower2_diagnoses': 'L2',
        'tower3_demographics': 'StandardScaler',
        'tower4_place': 'StandardScaler',
        'tower5_cost': 'StandardScaler',
        'tower6_pin': 'StandardScaler'
    },
    'missing_data_strategy': 'zero_padding',
    'alignment': {
        'me2vec_pins': len(all_pins_union),
        'linear_pins': len(all_pins_linear),
        'final_pins': len(final_pins),
        'pins_in_both_modalities': len(pins_in_both),
        'pins_only_procedures': len(pins_only_proc),
        'pins_only_diagnoses': len(pins_only_diag)
    }
}

with open('final_me2vec_all_towers_metadata.pkl', 'wb') as f:
    pickle.dump(final_metadata, f)

print(f"✓ Saved: final_me2vec_all_towers_metadata.pkl")

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nFinal embeddings: {all_embeddings.shape}")
print(f"\nTower breakdown:")
for tower, (start, end) in final_metadata['tower_ranges'].items():
    dims = end - start
    emb_slice = all_embeddings[:, start:end]
    norm_type = final_metadata['normalization'][tower]
    print(f"  {tower:25s}: dims [{start:4d}:{end:4d}] = {dims:4d} dims "
          f"({norm_type:15s}) mean={emb_slice.mean():.4f}, std={emb_slice.std():.4f}")

print("\n" + "="*80)
print("ALL ME2VEC TOWERS COMBINED - COMPLETE!")
print("="*80)
print(f"\nOutput files:")
print(f"  - {output_parquet}")
print(f"  - {output_npy}")
print(f"  - final_me2vec_all_towers_metadata.pkl")
print(f"  - me2vec_linear_towers_scalers.pkl")
print(f"\nReady for similarity analysis and validation!")
