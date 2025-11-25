import pandas as pd
import numpy as np
import random

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

embeddings_df = pd.read_parquet('final_me2vec_all_towers_1046d.parquet')
print(f"Embeddings: {embeddings_df.shape}")

procedure_df = pd.read_parquet('procedure_df.parquet')
print(f"Procedure data: {procedure_df.shape}")

diagnosis_df = pd.read_parquet('diag_grouped.parquet')
print(f"Diagnosis data: {diagnosis_df.shape}")

pin_names_df = pd.read_parquet('all_pin_names.parquet')
all_pins_with_embeddings = embeddings_df['PIN'].values
pin_names_df = pin_names_df[pin_names_df['PIN'].isin(all_pins_with_embeddings)]
print(f"PIN names: {pin_names_df.shape}")

prov_spl_df = pd.read_parquet('prov_spl.parquet')
county_df = pd.read_parquet('county_df.parquet')
code_desc_df = pd.read_parquet('code_desc_df.parquet')

print("\n" + "="*80)
print("PREPARING DATA STRUCTURES")
print("="*80)

pin_to_name = dict(zip(pin_names_df['PIN'], pin_names_df['PIN_name']))
pin_to_specialty = dict(zip(prov_spl_df['PIN'], prov_spl_df['srv_spclty_ctg_cd']))

county_df['county_state'] = county_df['county_nm'].fillna('') + '|' + county_df['state_postal_cd'].fillna('')
pin_to_county_state = dict(zip(county_df['PIN'], county_df['county_state']))

county_state_to_pins = {}
for pin, county_state in pin_to_county_state.items():
    if county_state not in county_state_to_pins:
        county_state_to_pins[county_state] = []
    county_state_to_pins[county_state].append(pin)

all_pins_list = embeddings_df['PIN'].values
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins_list)}

emb_cols = [col for col in embeddings_df.columns if col != 'PIN']
embeddings_matrix = embeddings_df[emb_cols].values
pin_to_emb_idx = {pin: idx for idx, pin in enumerate(all_pins_list)}

PROCEDURE_TOWER_START = 0
PROCEDURE_TOWER_END = 512
DIAGNOSIS_TOWER_START = 512
DIAGNOSIS_TOWER_END = 1024

procedure_tower_matrix = embeddings_matrix[:, PROCEDURE_TOWER_START:PROCEDURE_TOWER_END]
diagnosis_tower_matrix = embeddings_matrix[:, DIAGNOSIS_TOWER_START:DIAGNOSIS_TOWER_END]

procedure_summary = procedure_df.groupby('PIN').agg({
    'code': lambda x: set(x),
    'claims': lambda x: dict(zip(procedure_df.loc[x.index, 'code'], x))
}).reset_index()
procedure_summary.columns = ['PIN', 'codes', 'code_to_claims']
pin_to_procedure_codes = dict(zip(procedure_summary['PIN'], procedure_summary['codes']))
pin_to_procedure_claims = dict(zip(procedure_summary['PIN'], procedure_summary['code_to_claims']))

diagnosis_summary = diagnosis_df.groupby('PIN').agg({
    'code': lambda x: set(x),
    'claims': lambda x: dict(zip(diagnosis_df.loc[x.index, 'code'], x))
}).reset_index()
diagnosis_summary.columns = ['PIN', 'codes', 'code_to_claims']
pin_to_diagnosis_codes = dict(zip(diagnosis_summary['PIN'], diagnosis_summary['codes']))
pin_to_diagnosis_claims = dict(zip(diagnosis_summary['PIN'], diagnosis_summary['code_to_claims']))

def format_description(desc):
    if pd.isna(desc) or desc == '':
        return ''
    desc = str(desc).lower()
    sentences = desc.split('. ')
    formatted_sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    return '. '.join(formatted_sentences)

code_to_desc_raw = code_desc_df.groupby('code')['code_desc'].apply(lambda x: ', '.join(x.unique())).to_dict()
code_to_desc = {code: format_description(desc) for code, desc in code_to_desc_raw.items()}

print(f"Embeddings matrix: {embeddings_matrix.shape}")
print(f"Procedure tower matrix: {procedure_tower_matrix.shape}")
print(f"Diagnosis tower matrix: {diagnosis_tower_matrix.shape}")
print(f"Procedure codes mapped: {len(pin_to_procedure_codes)} providers")
print(f"Diagnosis codes mapped: {len(pin_to_diagnosis_codes)} providers")
print(f"Code descriptions: {len(code_to_desc)}")

print("\n" + "="*80)
print("SAMPLING HOSPITALS (10 WHOS)")
print("="*80)

procedure_code_counts = {pin: len(codes) for pin, codes in pin_to_procedure_codes.items()}

def count_alternatives_in_county(pin):
    county = pin_to_county_state.get(pin, '')
    same_county_pins = county_state_to_pins.get(county, [])
    return len([p for p in same_county_pins if p in pin_to_emb_idx and p != pin])

whos_pins = [pin for pin in all_pins_list 
             if pin_to_specialty.get(pin) == 'WHOS' 
             and count_alternatives_in_county(pin) >= 5
             and pin in procedure_code_counts]

whos_pins.sort(key=lambda p: procedure_code_counts[p], reverse=True)
top_whos = whos_pins[:max(1, len(whos_pins) // 5)]

random.seed(42)
selected_pins = random.sample(top_whos, min(10, len(top_whos)))

sampled_hospitals = []
for pin in selected_pins:
    sampled_hospitals.append({
        'pin': pin,
        'name': pin_to_name.get(pin, 'Unknown'),
        'specialty': pin_to_specialty.get(pin, 'Unknown')
    })

print(f"Sampled {len(sampled_hospitals)} WHOS hospitals")

def cosine_similarity_vectorized(vec, matrix):
    vec = vec.reshape(1, -1)
    dot_products = np.dot(matrix, vec.T).flatten()
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec)
    return dot_products / (norms + 1e-8)

def get_top_5_alternatives_same_county(primary_pin):
    if primary_pin not in pin_to_emb_idx:
        return []
    
    primary_county = pin_to_county_state.get(primary_pin, '')
    same_county_pins = county_state_to_pins.get(primary_county, [])
    candidate_pins = [p for p in same_county_pins if p in pin_to_emb_idx and p != primary_pin]
    
    if not candidate_pins:
        return []
    
    primary_emb = embeddings_matrix[pin_to_emb_idx[primary_pin]]
    candidate_indices = [pin_to_emb_idx[p] for p in candidate_pins]
    candidate_embs = embeddings_matrix[candidate_indices]
    
    similarities = cosine_similarity_vectorized(primary_emb, candidate_embs)
    top_5_indices = np.argsort(similarities)[-5:][::-1]
    
    return [candidate_pins[i] for i in top_5_indices]

def get_overall_similarity(primary_pin, alt_pin):
    if primary_pin not in pin_to_emb_idx or alt_pin not in pin_to_emb_idx:
        return 0.0
    primary_emb = embeddings_matrix[pin_to_emb_idx[primary_pin]]
    alt_emb = embeddings_matrix[pin_to_emb_idx[alt_pin]]
    dot = np.dot(primary_emb, alt_emb)
    norm = np.linalg.norm(primary_emb) * np.linalg.norm(alt_emb)
    return dot / (norm + 1e-8)

def get_tower_similarity(primary_pin, alt_pin, tower_matrix):
    if primary_pin not in pin_to_emb_idx or alt_pin not in pin_to_emb_idx:
        return 0.0
    primary_tower = tower_matrix[pin_to_emb_idx[primary_pin]]
    alt_tower = tower_matrix[pin_to_emb_idx[alt_pin]]
    dot = np.dot(primary_tower, alt_tower)
    norm = np.linalg.norm(primary_tower) * np.linalg.norm(alt_tower)
    return dot / (norm + 1e-8)

def compute_overlap_metrics(primary_pin, alt_pin, pin_to_codes, pin_to_claims):
    primary_codes = pin_to_codes.get(primary_pin, set())
    alt_codes = pin_to_codes.get(alt_pin, set())
    primary_claims = pin_to_claims.get(primary_pin, {})
    alt_claims = pin_to_claims.get(alt_pin, {})
    
    overlapping_codes = primary_codes & alt_codes
    non_overlapping_primary = primary_codes - alt_codes
    non_overlapping_alt = alt_codes - primary_codes
    
    overlapping_codes_count = len(overlapping_codes)
    non_overlapping_codes_primary = len(non_overlapping_primary)
    
    claims_overlapping = sum(primary_claims.get(c, 0) for c in overlapping_codes)
    claims_non_overlapping_primary = sum(primary_claims.get(c, 0) for c in non_overlapping_primary)
    claims_non_overlapping_alt = sum(alt_claims.get(c, 0) for c in non_overlapping_alt)
    
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

def generate_tables(sampled_hospitals, tower_matrix, pin_to_codes, pin_to_claims, tower_name):
    print(f"\n" + "="*80)
    print(f"BUILDING {tower_name.upper()} TABLES")
    print("="*80)
    
    table1_rows = []
    table2_rows = []
    
    for hospital in sampled_hospitals:
        primary_pin = hospital['pin']
        primary_name = hospital['name']
        
        top_5_alts = get_top_5_alternatives_same_county(primary_pin)
        
        if not top_5_alts:
            print(f"  {primary_pin}: No alternatives in same county, skipping")
            continue
        
        print(f"  {primary_pin}: Found {len(top_5_alts)} alternatives in same county")
        
        for alt_pin in top_5_alts:
            alt_name = pin_to_name.get(alt_pin, 'Unknown')
            overall_sim = get_overall_similarity(primary_pin, alt_pin)
            tower_sim = get_tower_similarity(primary_pin, alt_pin, tower_matrix)
            metrics = compute_overlap_metrics(primary_pin, alt_pin, pin_to_codes, pin_to_claims)
            
            table1_rows.append({
                'Primary_PIN': primary_pin,
                'Primary_Name': primary_name,
                'Alternative_PIN': alt_pin,
                'Alternative_Name': alt_name,
                'Overall_Similarity': round(overall_sim, 4),
                f'{tower_name}_Similarity': round(tower_sim, 4),
                'Overlapping_Codes_Count': metrics['overlapping_codes_count'],
                'NonOverlapping_Codes_Primary': metrics['non_overlapping_codes_primary'],
                'Claims_Overlapping': metrics['claims_overlapping'],
                'Claims_NonOverlapping_Primary': metrics['claims_non_overlapping_primary'],
                'Claims_NonOverlapping_Alternative': metrics['claims_non_overlapping_alt'],
                'Pct_Claims_NonOverlapping_Alternative': round(metrics['pct_claims_non_overlapping_alt'], 2)
            })
        
        primary_codes = pin_to_codes.get(primary_pin, set())
        primary_claims = pin_to_claims.get(primary_pin, {})
        primary_total = sum(primary_claims.values()) if primary_claims else 1
        
        alt_pins = list(top_5_alts)
        
        codes_sorted = sorted(primary_codes, key=lambda c: primary_claims.get(c, 0), reverse=True)
        
        alt_claims_list = []
        alt_totals = []
        for ap in alt_pins:
            claims = pin_to_claims.get(ap, {})
            alt_claims_list.append(claims)
            alt_totals.append(sum(claims.values()) if claims else 1)
        
        for code in codes_sorted:
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
            
            for i in range(5):
                if i < len(alt_pins):
                    row[f'Alt{i+1}_PIN'] = alt_pins[i]
                    row[f'Alt{i+1}_Claims'] = alt_claims_list[i].get(code, 0)
                    alt_total = alt_totals[i]
                    row[f'Alt{i+1}_Pct'] = round((alt_claims_list[i].get(code, 0) / alt_total * 100), 2) if alt_total > 0 else 0
                else:
                    row[f'Alt{i+1}_PIN'] = ''
                    row[f'Alt{i+1}_Claims'] = 0
                    row[f'Alt{i+1}_Pct'] = 0
            
            table2_rows.append(row)
    
    table1_df = pd.DataFrame(table1_rows)
    table2_df = pd.DataFrame(table2_rows)
    
    print(f"\nTable 1 shape: {table1_df.shape}")
    print(f"Table 2 shape: {table2_df.shape}")
    
    return table1_df, table2_df

procedure_table1_df, procedure_table2_df = generate_tables(
    sampled_hospitals, 
    procedure_tower_matrix, 
    pin_to_procedure_codes, 
    pin_to_procedure_claims, 
    'Procedure'
)

diagnosis_table1_df, diagnosis_table2_df = generate_tables(
    sampled_hospitals, 
    diagnosis_tower_matrix, 
    pin_to_diagnosis_codes, 
    pin_to_diagnosis_claims, 
    'Diagnosis'
)

print(f"\n" + "="*80)
print("SAVING OUTPUT FILES")
print("="*80)

procedure_table1_df.to_csv('procedure_table1_raw_data.csv', index=False)
procedure_table1_df.to_parquet('procedure_table1_raw_data.parquet', index=False)

procedure_table2_df.to_csv('procedure_table2_raw_data.csv', index=False)
procedure_table2_df.to_parquet('procedure_table2_raw_data.parquet', index=False)

diagnosis_table1_df.to_csv('diagnosis_table1_raw_data.csv', index=False)
diagnosis_table1_df.to_parquet('diagnosis_table1_raw_data.parquet', index=False)

diagnosis_table2_df.to_csv('diagnosis_table2_raw_data.csv', index=False)
diagnosis_table2_df.to_parquet('diagnosis_table2_raw_data.parquet', index=False)

print(f"\nProcedure Table 1: procedure_table1_raw_data.csv/parquet ({len(procedure_table1_df)} rows)")
print(f"Procedure Table 2: procedure_table2_raw_data.csv/parquet ({len(procedure_table2_df)} rows)")
print(f"Diagnosis Table 1: diagnosis_table1_raw_data.csv/parquet ({len(diagnosis_table1_df)} rows)")
print(f"Diagnosis Table 2: diagnosis_table2_raw_data.csv/parquet ({len(diagnosis_table2_df)} rows)")

print(f"\n" + "="*80)
print("PROCEDURE TABLE 1 SAMPLE:")
print("="*80)
print(procedure_table1_df.head(5).to_string())

print(f"\n" + "="*80)
print("PROCEDURE TABLE 2 SAMPLE:")
print("="*80)
print(procedure_table2_df.head(5).to_string())

print(f"\n" + "="*80)
print("DIAGNOSIS TABLE 1 SAMPLE:")
print("="*80)
print(diagnosis_table1_df.head(5).to_string())

print(f"\n" + "="*80)
print("DIAGNOSIS TABLE 2 SAMPLE:")
print("="*80)
print(diagnosis_table2_df.head(5).to_string())
