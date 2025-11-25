"""
PROCEDURE TABLE 1 - RAW DATA GENERATOR (COUNTY-LEVEL)
=====================================================

Creates raw data for 32 sampled providers with top 5 alternatives each.
Alternatives are filtered to SAME COUNTY only.
Ranking is by OVERALL SIMILARITY (no CNP).

Columns:
- Primary_PIN
- Primary_Name
- Alternative_PIN
- Alternative_Name
- Overall_Similarity
- Procedure_Similarity (procedure tower only)
- Overlapping_Codes_Count
- NonOverlapping_Codes_Primary
- Claims_Overlapping
- Claims_NonOverlapping_Primary
- Claims_NonOverlapping_Alternative
- Pct_Claims_NonOverlapping_Alternative

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import pickle
import random

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)
print(f"Labels: {len(pin_to_label)}")

embeddings_df = pd.read_parquet('final_me2vec_all_towers_1046d.parquet')
print(f"Embeddings: {embeddings_df.shape}")

procedure_df = pd.read_parquet('procedure_df.parquet')
print(f"Procedure data: {procedure_df.shape}")

pin_names_df = pd.read_parquet('all_pin_names.parquet')
all_pins_with_embeddings = embeddings_df['PIN'].values
pin_names_df = pin_names_df[pin_names_df['PIN'].isin(all_pins_with_embeddings)]
print(f"PIN names: {pin_names_df.shape}")

pin_to_name = dict(zip(pin_names_df['PIN'], pin_names_df['PIN_name']))

prov_spl_df = pd.read_parquet('prov_spl.parquet')
pin_to_specialty = dict(zip(prov_spl_df['PIN'], prov_spl_df['srv_spclty_ctg_cd']))

# Load county data
county_df = pd.read_parquet('county_df.parquet')
print(f"County data: {county_df.shape}")

county_df['county_state'] = county_df['county_nm'].fillna('') + '|' + county_df['state_postal_cd'].fillna('')
pin_to_county_state = dict(zip(county_df['PIN'], county_df['county_state']))
print(f"PIN to county mapping: {len(pin_to_county_state)} providers")

# Build county_state to PINs mapping
county_state_to_pins = {}
for pin, county_state in pin_to_county_state.items():
    if county_state not in county_state_to_pins:
        county_state_to_pins[county_state] = []
    county_state_to_pins[county_state].append(pin)

print(f"Unique county+state combinations: {len(county_state_to_pins)}")

# =============================================================================
# PREPARE DATA STRUCTURES
# =============================================================================

print("\n" + "="*80)
print("PREPARING DATA STRUCTURES")
print("="*80)

all_pins_list = embeddings_df['PIN'].values.tolist()
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins_list)}

emb_cols = [col for col in embeddings_df.columns if col != 'PIN']
pin_to_emb = dict(zip(embeddings_df['PIN'], embeddings_df[emb_cols].values.tolist()))

# Procedure tower dimensions
PROCEDURE_TOWER_START = 0
PROCEDURE_TOWER_END = 512

# Process procedure data
procedure_summary = procedure_df.groupby('PIN').agg({
    'code': lambda x: set(x),
    'claims': lambda x: dict(zip(procedure_df.loc[x.index, 'code'], x))
}).reset_index()
procedure_summary.columns = ['PIN', 'codes', 'code_to_claims']
pin_to_procedure_codes = dict(zip(procedure_summary['PIN'], procedure_summary['codes']))
pin_to_procedure_claims = dict(zip(procedure_summary['PIN'], procedure_summary['code_to_claims']))

print(f"Procedure codes mapped: {len(pin_to_procedure_codes)} providers")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cosine_similarity(vec_a, vec_b):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b + 1e-8)

def get_overall_similarity(pin_a, pin_b):
    """Full 1046-dim embedding similarity."""
    emb_a = pin_to_emb.get(pin_a)
    emb_b = pin_to_emb.get(pin_b)
    if emb_a is None or emb_b is None:
        return 0.0
    return cosine_similarity(emb_a, emb_b)

def get_procedure_similarity(pin_a, pin_b):
    """Procedure tower only (dims 0-512)."""
    emb_a = pin_to_emb.get(pin_a)
    emb_b = pin_to_emb.get(pin_b)
    if emb_a is None or emb_b is None:
        return 0.0
    tower_a = np.array(emb_a)[PROCEDURE_TOWER_START:PROCEDURE_TOWER_END]
    tower_b = np.array(emb_b)[PROCEDURE_TOWER_START:PROCEDURE_TOWER_END]
    return cosine_similarity(tower_a, tower_b)

def get_top_5_alternatives_same_county(primary_pin):
    """Get top 5 alternatives from SAME COUNTY based on OVERALL similarity."""
    if primary_pin not in pin_to_idx:
        return []
    
    primary_emb = pin_to_emb.get(primary_pin)
    if primary_emb is None:
        return []
    
    # Get primary's county
    primary_county = pin_to_county_state.get(primary_pin, '')
    
    # Get all PINs in same county
    same_county_pins = county_state_to_pins.get(primary_county, [])
    
    # Filter to those with embeddings, exclude primary
    candidate_pins = [p for p in same_county_pins if p in pin_to_emb and p != primary_pin]
    
    if not candidate_pins:
        return []
    
    # Calculate similarities
    similarities = []
    for alt_pin in candidate_pins:
        alt_emb = pin_to_emb.get(alt_pin)
        if alt_emb is None:
            continue
        overall_sim = cosine_similarity(primary_emb, alt_emb)
        similarities.append((alt_pin, overall_sim))
    
    # Sort by overall similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 5 PINs
    return [pin for pin, _ in similarities[:5]]

def compute_overlap_metrics(primary_pin, alt_pin):
    """Compute all overlap metrics between primary and alternative."""
    primary_codes = pin_to_procedure_codes.get(primary_pin, set())
    alt_codes = pin_to_procedure_codes.get(alt_pin, set())
    primary_claims = pin_to_procedure_claims.get(primary_pin, {})
    alt_claims = pin_to_procedure_claims.get(alt_pin, {})
    
    overlapping_codes = primary_codes & alt_codes
    non_overlapping_primary = primary_codes - alt_codes
    non_overlapping_alt = alt_codes - primary_codes
    
    # Counts
    overlapping_codes_count = len(overlapping_codes)
    non_overlapping_codes_primary = len(non_overlapping_primary)
    
    # Claims from overlapping codes (primary side)
    claims_overlapping = sum(primary_claims.get(c, 0) for c in overlapping_codes)
    
    # Claims from non-overlapping codes (primary side)
    claims_non_overlapping_primary = sum(primary_claims.get(c, 0) for c in non_overlapping_primary)
    
    # Claims from non-overlapping codes (alternative side)
    claims_non_overlapping_alt = sum(alt_claims.get(c, 0) for c in non_overlapping_alt)
    
    # % of alternative's total claims from non-overlapping codes
    alt_total_claims = sum(alt_claims.values())
    pct_claims_non_overlapping_alt = (claims_non_overlapping_alt / alt_total_claims * 100) if alt_total_claims > 0 else 0
    
    return {
        'overlapping_codes_count': overlapping_codes_count,
        'non_overlapping_codes_primary': non_overlapping_codes_primary,
        'claims_overlapping': claims_overlapping,
        'claims_non_overlapping_primary': claims_non_overlapping_primary,
        'claims_non_overlapping_alt': claims_non_overlapping_alt,
        'pct_claims_non_overlapping_alt': pct_claims_non_overlapping_alt
    }

# =============================================================================
# SAMPLE HOSPITALS (WHOS + UNLABELED ONLY, TOTAL 10)
# =============================================================================

print("\n" + "="*80)
print("SAMPLING HOSPITALS (WHOS + UNLABELED ONLY)")
print("="*80)

procedure_code_counts = {pin: len(codes) for pin, codes in pin_to_procedure_codes.items()}

# Get WHOS labeled hospitals
whos_pins = [pin for pin, label in pin_to_label.items() if label == 'WHOS' and pin in pin_to_idx]
print(f"WHOS hospitals: {len(whos_pins)}")

# Get unlabeled hospitals
unlabeled_pins = [pin for pin in all_pins_list if pin not in pin_to_label]
print(f"Unlabeled hospitals: {len(unlabeled_pins)}")

sampled_hospitals = []
random.seed(42)

# Sample 5 from WHOS (high procedure count)
whos_with_procedures = [p for p in whos_pins if p in procedure_code_counts]
whos_with_procedures.sort(key=lambda p: procedure_code_counts.get(p, 0), reverse=True)
top_whos = whos_with_procedures[:max(1, len(whos_with_procedures) // 5)]  # Top 20%

if len(top_whos) >= 5:
    selected_whos = random.sample(top_whos, 5)
else:
    selected_whos = top_whos[:5] if len(top_whos) >= 5 else top_whos

for pin in selected_whos:
    sampled_hospitals.append({
        'pin': pin,
        'label': 'WHOS',
        'name': pin_to_name.get(pin, 'Unknown'),
        'specialty': pin_to_specialty.get(pin, 'Unknown')
    })

print(f"Sampled {len(selected_whos)} WHOS hospitals")

# Sample 5 from unlabeled (high procedure count)
unlabeled_with_procedures = [p for p in unlabeled_pins if p in procedure_code_counts]
unlabeled_with_procedures.sort(key=lambda p: procedure_code_counts.get(p, 0), reverse=True)
top_unlabeled = unlabeled_with_procedures[:max(1, len(unlabeled_with_procedures) // 5)]  # Top 20%

if len(top_unlabeled) >= 5:
    selected_unlabeled = random.sample(top_unlabeled, 5)
else:
    selected_unlabeled = top_unlabeled[:5] if len(top_unlabeled) >= 5 else top_unlabeled

for pin in selected_unlabeled:
    sampled_hospitals.append({
        'pin': pin,
        'label': 'unlabeled',
        'name': pin_to_name.get(pin, 'Unknown'),
        'specialty': pin_to_specialty.get(pin, 'Unknown')
    })

print(f"Sampled {len(selected_unlabeled)} unlabeled hospitals")
print(f"Total sampled: {len(sampled_hospitals)} hospitals")

# =============================================================================
# BUILD TABLE 1 AND TABLE 2 RAW DATA
# =============================================================================

print("\n" + "="*80)
print("BUILDING TABLE 1 AND TABLE 2 RAW DATA")
print("="*80)

# Load code descriptions
code_desc_df = pd.read_parquet('code_desc_grouped.parquet')
code_to_desc = (
    code_desc_df
    .groupby('code')['code_desc']
    .apply(lambda x: ', '.join(x.unique()))
    .to_dict()
)
print(f"Code descriptions loaded: {len(code_to_desc)}")

table1_rows = []
table2_rows = []

for hospital in sampled_hospitals:
    primary_pin = hospital['pin']
    primary_name = hospital['name']
    
    # Get top 5 alternatives from SAME COUNTY
    top_5_alts = get_top_5_alternatives_same_county(primary_pin)
    
    if not top_5_alts:
        print(f"  {primary_pin}: No alternatives in same county, skipping")
        continue
    
    print(f"  {primary_pin}: Found {len(top_5_alts)} alternatives in same county")
    
    # ----- TABLE 1: Alternatives Summary -----
    for alt_pin in top_5_alts:
        alt_name = pin_to_name.get(alt_pin, 'Unknown')
        
        overall_sim = get_overall_similarity(primary_pin, alt_pin)
        procedure_sim = get_procedure_similarity(primary_pin, alt_pin)
        metrics = compute_overlap_metrics(primary_pin, alt_pin)
        
        table1_rows.append({
            'Primary_PIN': primary_pin,
            'Primary_Name': primary_name,
            'Alternative_PIN': alt_pin,
            'Alternative_Name': alt_name,
            'Overall_Similarity': round(overall_sim, 4),
            'Procedure_Similarity': round(procedure_sim, 4),
            'Overlapping_Codes_Count': metrics['overlapping_codes_count'],
            'NonOverlapping_Codes_Primary': metrics['non_overlapping_codes_primary'],
            'Claims_Overlapping': metrics['claims_overlapping'],
            'Claims_NonOverlapping_Primary': metrics['claims_non_overlapping_primary'],
            'Claims_NonOverlapping_Alternative': metrics['claims_non_overlapping_alt'],
            'Pct_Claims_NonOverlapping_Alternative': round(metrics['pct_claims_non_overlapping_alt'], 2)
        })
    
    # ----- TABLE 2: Code-Level Drilldown -----
    primary_codes = pin_to_procedure_codes.get(primary_pin, set())
    primary_claims = pin_to_procedure_claims.get(primary_pin, {})
    primary_total = sum(primary_claims.values()) if primary_claims else 1
    
    # Get all codes from primary and alternatives
    all_codes = set(primary_codes)
    for alt_pin in top_5_alts:
        all_codes.update(pin_to_procedure_codes.get(alt_pin, set()))
    
    # Sort by primary claims descending, limit to top 50
    all_codes = sorted(all_codes, key=lambda c: primary_claims.get(c, 0), reverse=True)[:50]
    
    # Get alternative claims data
    alt_claims_list = []
    alt_totals = []
    for alt_pin in top_5_alts:
        claims = pin_to_procedure_claims.get(alt_pin, {})
        alt_claims_list.append(claims)
        alt_totals.append(sum(claims.values()) if claims else 1)
    
    for code in all_codes:
        desc = code_to_desc.get(code, code)
        if len(str(desc)) > 100:
            desc = str(desc)[:100]
        
        primary_claim_count = primary_claims.get(code, 0)
        primary_pct = round((primary_claim_count / primary_total * 100), 2) if primary_total > 0 else 0
        
        row = {
            'Primary_PIN': primary_pin,
            'Primary_Name': primary_name,
            'Code': code,
            'Description': desc,
            'Primary_Claims': primary_claim_count,
            'Primary_Pct': primary_pct
        }
        
        # Add each alternative's data
        for i, alt_pin in enumerate(top_5_alts, 1):
            alt_claim_count = alt_claims_list[i-1].get(code, 0)
            alt_pct = round((alt_claim_count / alt_totals[i-1] * 100), 2) if alt_totals[i-1] > 0 else 0
            row[f'Alt{i}_PIN'] = alt_pin
            row[f'Alt{i}_Claims'] = alt_claim_count
            row[f'Alt{i}_Pct'] = alt_pct
        
        # Fill missing alternatives with empty values
        for i in range(len(top_5_alts) + 1, 6):
            row[f'Alt{i}_PIN'] = ''
            row[f'Alt{i}_Claims'] = 0
            row[f'Alt{i}_Pct'] = 0
        
        table2_rows.append(row)

table1_df = pd.DataFrame(table1_rows)
table2_df = pd.DataFrame(table2_rows)

print(f"\nTable 1 shape: {table1_df.shape}")
print(f"Table 2 shape: {table2_df.shape}")

# =============================================================================
# SAVE TO CSV AND PARQUET
# =============================================================================

table1_df.to_csv('procedure_table1_raw_data.csv', index=False)
table1_df.to_parquet('procedure_table1_raw_data.parquet', index=False)

table2_df.to_csv('procedure_table2_raw_data.csv', index=False)
table2_df.to_parquet('procedure_table2_raw_data.parquet', index=False)

print(f"\n" + "="*80)
print("OUTPUT FILES SAVED")
print("="*80)
print(f"\nTable 1 (Alternatives Summary):")
print(f"  - procedure_table1_raw_data.csv")
print(f"  - procedure_table1_raw_data.parquet")
print(f"  - Rows: {len(table1_df)}")

print(f"\nTable 2 (Code-Level Drilldown):")
print(f"  - procedure_table2_raw_data.csv")
print(f"  - procedure_table2_raw_data.parquet")
print(f"  - Rows: {len(table2_df)}")

print(f"\n" + "="*80)
print("TABLE 1 SAMPLE:")
print("="*80)
print(table1_df.head(5).to_string())

print(f"\n" + "="*80)
print("TABLE 2 SAMPLE:")
print("="*80)
print(table2_df.head(5).to_string())
