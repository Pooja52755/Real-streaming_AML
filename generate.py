import os
import hashlib
import pandas as pd
import numpy as np

def generate_accounts_file(
    trans_csv_path: str = "Data/HI-Small_FANOUT_10M_test.csv",
    output_accounts_path: str = "Data/HI-Small_FANOUT_10M_accounts.csv",
    aux_files: list = None
):
    """
    Generates a comprehensive, rich Accounts metadata CSV for the AML dataset.
    Extracts all unique accounts from the transaction dataset (and auxiliary testing datasets),
    and deterministically generates realistic banking metadata, KYC status, locations,
    entity classifications, and opening dates.
    """
    if aux_files is None:
        aux_files = ["Data/testing_accounts.csv", "Data/testing_trans.csv"]

    print(f"Reading main transaction dataset from: {trans_csv_path}...")
    
    usecols = ['From Bank', 'Account', 'To Bank', 'Account.1']
    df_main = pd.read_csv(trans_csv_path, usecols=usecols)
    print(f"Loaded {len(df_main):,} main transactions.")

    dfs_to_combine = [df_main]
    
    # Read any auxiliary datasets if available
    for aux_path in aux_files:
        if os.path.exists(aux_path):
            try:
                aux_df = pd.read_csv(aux_path)
                if 'From Bank' in aux_df.columns and 'Account' in aux_df.columns:
                    dfs_to_combine.append(aux_df[['From Bank', 'Account', 'To Bank', 'Account.1']])
                    print(f"Included auxiliary transaction dataset: {aux_path} ({len(aux_df)} rows)")
            except Exception as e:
                print(f"Skipping aux file {aux_path}: {e}")

    # Combine all transaction frames
    df_all_trans = pd.concat(dfs_to_combine, ignore_index=True)

    # Extract Sender Accounts
    df_senders = df_all_trans[['From Bank', 'Account']].rename(
        columns={'From Bank': 'Bank ID', 'Account': 'Account Number'}
    )
    
    # Extract Receiver Accounts
    df_receivers = df_all_trans[['To Bank', 'Account.1']].rename(
        columns={'To Bank': 'Bank ID', 'Account.1': 'Account Number'}
    )
    
    # Combine all account occurrences
    all_accounts = pd.concat([df_senders, df_receivers], ignore_index=True)
    all_accounts['Bank ID'] = all_accounts['Bank ID'].astype(str).str.strip()
    all_accounts['Account Number'] = all_accounts['Account Number'].astype(str).str.strip()
    
    # Get unique (Account Number, Bank ID) pairs
    unique_accs = all_accounts[['Account Number', 'Bank ID']].drop_duplicates(subset=['Account Number']).reset_index(drop=True)
    print(f"Found {len(unique_accs):,} total unique accounts.")
    
    # Check if existing metadata file (e.g. testing_trans.csv) has preset entity names
    preset_meta = {}
    if os.path.exists("Data/testing_trans.csv"):
        try:
            df_preset = pd.read_csv("Data/testing_trans.csv")
            if 'Account Number' in df_preset.columns:
                for _, r in df_preset.iterrows():
                    acc_key = str(r.get('Account Number', '')).strip()
                    if acc_key:
                        preset_meta[acc_key] = {
                            'Bank Name': r.get('Bank Name'),
                            'Entity ID': r.get('Entity ID'),
                            'Entity Name': r.get('Entity Name')
                        }
        except Exception:
            pass

    # Deterministic metadata generator based on account hash
    countries_cities = [
        "New York, USA", "London, UK", "Frankfurt, Germany", "Zurich, Switzerland",
        "Tokyo, Japan", "Singapore", "Toronto, Canada", "Sydney, Australia",
        "Paris, France", "Madrid, Spain", "Dubai, UAE", "Hong Kong"
    ]
    
    entity_types = ["Corporation", "Sole Proprietorship", "Partnership", "Limited Liability Company", "Individual"]
    num_cities = len(countries_cities)
    num_etypes = len(entity_types)

    bank_names = []
    entity_ids = []
    entity_names = []
    account_types = []
    kyc_statuses = []
    cities = []
    open_dates = []
    
    acc_list = unique_accs['Account Number'].tolist()
    bank_id_list = unique_accs['Bank ID'].tolist()

    for acc, bank_id in zip(acc_list, bank_id_list):
        # Stable hash across Python runs
        h = int(hashlib.md5(acc.encode('utf-8')).hexdigest(), 16)
        preset = preset_meta.get(acc)
        
        # Bank Name
        if preset and pd.notna(preset.get('Bank Name')):
            b_name = str(preset['Bank Name'])
        else:
            b_name = f"Global Bank #{bank_id}"
        bank_names.append(b_name)
        
        # Entity ID & Type
        if preset and pd.notna(preset.get('Entity ID')):
            e_id = str(preset['Entity ID'])
        else:
            e_id = f"ENT-{acc[:8]}"
        entity_ids.append(e_id)

        if preset and pd.notna(preset.get('Entity Name')):
            e_name = str(preset['Entity Name'])
            e_type = "Corporate" if ("Corp" in e_name or "LLC" in e_name or "Part" in e_name or "Sole" in e_name) else "Individual"
        else:
            e_type = entity_types[h % num_etypes]
            e_num = (h % 90000) + 10000
            e_name = f"{e_type} #{e_num}"
        entity_names.append(e_name)
        
        # Account Type & KYC
        acc_type = "Corporate Checking" if e_type != "Individual" else "Standard Checking"
        account_types.append(acc_type)
        kyc_statuses.append("Verified KYC")
        
        # City
        city = countries_cities[(h // 10) % num_cities]
        cities.append(city)
        
        # Open Date (deterministic 2018-2021 date)
        yr = 2018 + (h % 4)
        mo = (h % 12) + 1
        dy = (h % 28) + 1
        open_dates.append(f"{yr}-{mo:02d}-{dy:02d}")
        
    unique_accs['Bank Name'] = bank_names
    unique_accs['Entity ID'] = entity_ids
    unique_accs['Entity Name'] = entity_names
    unique_accs['Account Type'] = account_types
    unique_accs['KYC Status'] = kyc_statuses
    unique_accs['City'] = cities
    unique_accs['Open Date'] = open_dates
    
    # Reorder columns to match standard schema
    ordered_cols = [
        'Bank Name', 'Bank ID', 'Account Number', 'Entity ID', 'Entity Name',
        'Account Type', 'KYC Status', 'City', 'Open Date'
    ]
    unique_accs = unique_accs[ordered_cols]
    
    os.makedirs(os.path.dirname(output_accounts_path) or ".", exist_ok=True)
    unique_accs.to_csv(output_accounts_path, index=False)
    print(f"[SUCCESS] Successfully saved accounts metadata file ({len(unique_accs):,} accounts) to: {output_accounts_path}")
    return unique_accs

if __name__ == "__main__":
    generate_accounts_file()

