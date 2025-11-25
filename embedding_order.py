import pickle
with open('specialty_code_mappings.pkl', 'rb') as f:
   spec = pickle.load(f)
print("Keys:", spec.keys())
sample_indices = list(spec['code_indices'].values())[0][:5]
print("Sample code_indices:", sample_indices)
print("Type:", type(sample_indices[0]))


import pandas as pd
proc_df = pd.read_parquet('procedure_df.parquet')
print("Sample codes:", proc_df['code'].unique()[:10])
print("Code type:", type(proc_df['code'].iloc[0]))
