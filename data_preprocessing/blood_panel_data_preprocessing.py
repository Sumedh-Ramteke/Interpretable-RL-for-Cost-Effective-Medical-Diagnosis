import numpy as np
import pandas as pd
import math
from sklearn import preprocessing

def sepsis_data():
    # 1. Load Data
    filename = "datasets/sm_ddpo_sepsis_dataset.csv"
    # Try current dir or data/ subdir
    try:
        df = pd.read_csv(filename, na_values=["NULL"])
    except FileNotFoundError:
        df = pd.read_csv(f"data/{filename}", na_values=["NULL"])

    # 2. Handle Target & ID
    if 'hospital_expire_flag' in df.columns:
        y = df['hospital_expire_flag'].fillna(0).to_numpy().astype(np.int64)
        del df['hospital_expire_flag']
    else:
        raise ValueError("Target 'hospital_expire_flag' not found.")

    if 'icustay_id' in df.columns:
        del df['icustay_id']

    # 3. Handle Nominal Columns (One-Hot Encoding)
    # Identify categorical columns (object type)
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        # Create dummy variables (e.g., gender_M, gender_F)
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
        
    # 4. Handle Missing Values (CRITICAL STEP)
    # The RL environment needs a complete "Ground Truth" to function.
    # We fill NaNs with the column mean.
    df = df.fillna(df.mean())

    # 5. Define Feature Blocks (Dynamic Mapping)
    # Block 0: Demographics/History (Always Observed)
    block0_candidates = [
        'age', 'sofa_score', 'bmi', 
        'metastatic_cancer', 'diabetes', 'mechanical_ventilation'
    ]
    # Add generated dummy columns to Block 0 (e.g., 'gender_M', 'insurance_Private')
    # We check if the column starts with original categorical names or is a binary race col
    block0_prefixes = ['gender', 'race', 'marital', 'insurance', 'admission']
    
    # Identify all columns that belong to Block 0
    block0_cols = []
    for c in df.columns:
        # Check explicit candidates
        if c in block0_candidates:
            block0_cols.append(c)
        # Check prefixes (for one-hot encoded cols and race)
        elif any(c.startswith(p) for p in block0_prefixes):
            block0_cols.append(c)

    # All other columns are Tests
    test_cols = [c for c in df.columns if c not in block0_cols]

    # 6. Standardization & Block Construction
    # Process Block 0
    X_b0 = df[block0_cols].to_numpy().astype(np.float32)
    scaler0 = preprocessing.StandardScaler()
    X_b0 = scaler0.fit_transform(X_b0)
    
    # Process Test Blocks
    # Define Panels
    panel_vitals = ['heart_rate', 'resp_rate', 'sbp', 'dbp', 'map', 'spo2', 'temperature', 'gcs', 'urine_output', 'tidal_volume']
    panel_cbc = ['hemoglobin', 'hematocrit', 'platelet', 'rbc', 'wbc']
    panel_cmp = ['glucose', 'bicarbonate', 'creatinine', 'chloride', 'co2', 'sodium', 'potassium', 'bun', 'calcium']
    panel_liver = ['ast', 'bilirubin', 'albumin', 'magnesium']
    panel_gas = ['lactate', 'base_excess', 'ph', 'fio2', 'ptt', 'inr']
    all_panels = [panel_vitals, panel_cbc, panel_cmp, panel_liver, panel_gas]
    
    X_list = [X_b0]
    block = {}
    block[0] = np.arange(X_b0.shape[1])
    current_idx = X_b0.shape[1]
    block_id = 1
    
    used_tests = set()

    for panel in all_panels:
        valid_cols = [c for c in panel if c in df.columns]
        if not valid_cols: continue
        used_tests.update(valid_cols)
        
        X_panel = df[valid_cols].to_numpy().astype(np.float32)
        
        # Standardize (Now safe because NaNs are gone)
        scaler = preprocessing.StandardScaler()
        X_panel = scaler.fit_transform(X_panel)
        X_panel = np.clip(X_panel, -5, 5) # Clip outliers
        
        X_list.append(X_panel)
        block[block_id] = np.arange(current_idx, current_idx + X_panel.shape[1])
        current_idx += X_panel.shape[1]
        block_id += 1

    # Catch any remaining tests not in panels
    orphans = [c for c in test_cols if c not in used_tests]
    if orphans:
        X_orphan = df[orphans].to_numpy().astype(np.float32)
        X_orphan = preprocessing.StandardScaler().fit_transform(X_orphan)
        X_orphan = np.clip(X_orphan, -5, 5)
        X_list.append(X_orphan)
        block[block_id] = np.arange(current_idx, current_idx + X_orphan.shape[1])
    
    # Concatenate
    X_final = np.concatenate(X_list, axis=1)
    data = np.concatenate((X_final, y.reshape(len(y), 1)), axis=1)

    # 7. Costs (Relative weights)
    raw_costs = []
    
    for b_id in range(1, len(block)):
        if b_id == 1: 
            # Vitals (Heart rate, temp, etc.) - Usually observed or very cheap
            raw_costs.append(0) 
        elif b_id == 2:
            # CBC (Hemoglobin, WBC, etc.)
            raw_costs.append(44.0)
        elif b_id == 3:
            # CMP (Glucose, Creatinine, Electrolytes)
            raw_costs.append(48.0)
        elif b_id == 4:
            # Liver (Bilirubin, AST) - Often part of CMP, but we treat as separate
            raw_costs.append(48.0) 
        elif b_id == 5:
            # Gases & Coagulation (Lactate, PTT, INR)
            # The paper lists APTT as $473 and ABG as $26. 
            # We average them or take the max since we grouped them.
            raw_costs.append(473.0) 
        
    cost = np.array(raw_costs)    

    if len(cost) > 0:
        cost = cost / np.sum(cost) * len(cost)

    print(f"Data Processed. Shape: {data.shape}. Blocks: {len(block)}")
    return data, block, cost

# print(sepsis_data())

def aki_data():
    '''
    AKI Dataset Loader for SM-DDPO.
    
    Aligns with Table 4 of Yu et al. (2023):
    - Block 0: Observed ($0)   -> Demographics, Vitals, Urine, Vent
    - Block 1: CBC ($44)       -> Hgb, WBC, Platelets
    - Block 2: CMP ($48)       -> Glu, HCO3, Cr, BUN, Ca, K, eGFR
    - Block 3: APTT ($26)      -> PTT, INR, PT
    - Block 4: ABG ($473)      -> SpO2, pH, pO2, pCO2
    '''
    filename = "datasets/aki_dataset.csv"
    try:
        df = pd.read_csv(filename, na_values=["NULL"])
    except FileNotFoundError:
        df = pd.read_csv(f"data/{filename}", na_values=["NULL"])

    # 1. Handle Label
    if 'label' in df.columns:
        y = df['label'].fillna(0).to_numpy().astype(np.int64)
        del df['label']
    else:
        raise ValueError("Target 'label' not found.")
    
    if 'icustay_id' in df.columns: del df['icustay_id']

    # 2. Pre-process for eGFR (MDRD Equation)
    df['gender_num'] = df['gender'].apply(lambda x: 1 if x == 'M' else 0)
    df['race_black'] = df['race'].apply(lambda x: 1 if str(x).strip() == 'Black' else 0)
    
    # Calculate eGFR
    scr = df['creatinine_max'].clip(lower=0.1)
    age = df['age']
    df['egfr'] = 175 * (scr ** -1.154) * (age ** -0.203)
    df.loc[df['gender_num'] == 0, 'egfr'] *= 0.742 # Female adjustment
    df.loc[df['race_black'] == 1, 'egfr'] *= 1.212 # Black adjustment
    
    # 3. Impute Missing Values (Ground Truth Creation)
    df = df.fillna(df.mean(numeric_only=True))

    # =========================================================================
    # 4. Define Test Panels (Exact Match)
    # =========================================================================

    # Block 0: Observed ($0)
    panel_observed = [
        'age', 'gender_num', 'race_black',
        'heartrate_mean', 'heartrate_max',
        'sysbp_mean', 'sysbp_min', 'diasbp_mean', 'diasbp_min',
        'resprate_mean', 'tempc_max',
        'urine_output_total', 'mech_vent'
    ]

    # Block 1: CBC ($44)
    panel_cbc = ['hemoglobin_min', 'wbc_max', 'platelet_min']

    # Block 2: CMP ($48)
    panel_cmp = [
        'glucose_max', 'bicarbonate_min', 'creatinine_min', 'creatinine_max', 
        'bun_max', 'calcium_min', 'potassium_max', 'egfr'
    ]

    # Block 3: APTT ($26)
    panel_aptt = [
        'ptt_min', 'ptt_max', 'inr_min', 'inr_max', 'pt_min', 'pt_max'
    ]

    # Block 4: ABG ($473)
    panel_abg = [
        'spo2_min', 'spo2_max', 'ph_min', 'ph_max', 
        'pco2_min', 'pco2_max', 'po2_min', 'po2_max'
    ]

    all_panels = [panel_cbc, panel_cmp, panel_aptt, panel_abg]
    panel_costs_ref = [44.0, 48.0, 26.0, 473.0]

    # =========================================================================
    # 5. Build Data Matrices
    # =========================================================================
    X_list = []
    block = {}
    
    # Process Observed (Block 0)
    valid_b0 = [c for c in panel_observed if c in df.columns]
    X_b0 = df[valid_b0].to_numpy().astype(np.float32)
    X_b0 = preprocessing.StandardScaler().fit_transform(X_b0)
    X_b0 = np.clip(X_b0, -5, 5)  # Clip outliers (e.g. urine_output_total)
    
    X_list.append(X_b0)
    block[0] = np.arange(X_b0.shape[1])
    current_idx = X_b0.shape[1]
    
    # Process Panels (Blocks 1-4)
    block_id = 1
    raw_costs = []
    
    for i, panel in enumerate(all_panels):
        valid_cols = [c for c in panel if c in df.columns]
        if not valid_cols: continue
            
        X_panel = df[valid_cols].to_numpy().astype(np.float32)
        X_panel = preprocessing.StandardScaler().fit_transform(X_panel)
        X_panel = np.clip(X_panel, -5, 5) # Clip outliers
        
        X_list.append(X_panel)
        block[block_id] = np.arange(current_idx, current_idx + X_panel.shape[1])
        current_idx += X_panel.shape[1]
        raw_costs.append(panel_costs_ref[i])
        block_id += 1

    # Concatenate
    X_final = np.concatenate(X_list, axis=1)
    data = np.concatenate((X_final, y.reshape(len(y), 1)), axis=1)

    # Normalize Costs
    cost = np.array(raw_costs)
    if len(cost) > 0:
        cost = cost / np.sum(cost) * len(cost)

    print(f"AKI Data Processed. Shape: {data.shape}")
    print(f"Blocks: {len(block)} (0=Obs, 1=CBC, 2=CMP, 3=APTT, 4=ABG)")
    
    return data, block, cost

def ferritin_data():
    '''
    Proxy Ferritin Dataset Loader for SM-DDPO.
    
    Panels:
    - Block 0: Observed ($0)   -> Age, Gender
    - Block 1: CBC ($44)       -> Hgb, WBC, Platelets
    - Block 2: CMP ($48)       -> Cr, BUN, Glucose, Na, K, Cl, Ca
    - Block 3: Liver ($48)     -> Bilirubin, Albumin
    '''
    filename = "datasets/ferritin_dataset.csv"
    try:
        df = pd.read_csv(filename, na_values=["NULL"])
    except FileNotFoundError:
        df = pd.read_csv(f"data/{filename}", na_values=["NULL"])

    # 1. Handle Target & IDs
    if 'label' in df.columns:
        y = df['label'].fillna(0).to_numpy().astype(np.int64)
        del df['label']
    else:
        raise ValueError("Target 'label' not found.")
        
    if 'subject_id' in df.columns: del df['subject_id']
    if 'hadm_id' in df.columns: del df['hadm_id']

    # 2. Pre-processing Demographics
    if 'gender' in df.columns:
        df['gender_num'] = df['gender'].apply(lambda x: 1 if x == 'M' else 0)

    # 3. Handle Missing Values
    df = df.fillna(df.mean(numeric_only=True))

    # =========================================================================
    # 4. Define Feature Blocks
    # =========================================================================

    # Block 0: Observed (Cost $0)
    panel_observed = ['age', 'gender_num']

    # Block 1: CBC (Cost $44)
    panel_cbc = ['hemoglobin', 'wbc', 'platelets']

    # Block 2: CMP (Cost $48)
    panel_cmp = ['creatinine', 'bun', 'glucose', 'sodium', 'potassium', 'chloride', 'calcium']

    # Block 3: Liver Panel (Cost $48)
    panel_liver = ['bilirubin', 'albumin']

    all_panels = [panel_cbc, panel_cmp, panel_liver]
    panel_costs_ref = [44.0, 48.0, 48.0]

    # =========================================================================
    # 5. Build Matrices
    # =========================================================================
    X_list = []
    block = {}
    
    # Process Observed
    valid_b0 = [c for c in panel_observed if c in df.columns]
    X_b0 = df[valid_b0].to_numpy().astype(np.float32)
    X_b0 = preprocessing.StandardScaler().fit_transform(X_b0)
    
    X_list.append(X_b0)
    block[0] = np.arange(X_b0.shape[1])
    current_idx = X_b0.shape[1]
    
    # Process Test Panels
    block_id = 1
    raw_costs = []
    
    for i, panel in enumerate(all_panels):
        valid_cols = [c for c in panel if c in df.columns]
        if not valid_cols: continue
            
        X_panel = df[valid_cols].to_numpy().astype(np.float32)
        X_panel = preprocessing.StandardScaler().fit_transform(X_panel)
        X_panel = np.clip(X_panel, -5, 5) # Clip outliers
        
        X_list.append(X_panel)
        block[block_id] = np.arange(current_idx, current_idx + X_panel.shape[1])
        current_idx += X_panel.shape[1]
        raw_costs.append(panel_costs_ref[i])
        block_id += 1

    X_final = np.concatenate(X_list, axis=1)
    data = np.concatenate((X_final, y.reshape(len(y), 1)), axis=1)

    # Normalize Costs
    cost = np.array(raw_costs)
    if len(cost) > 0:
        cost = cost / np.sum(cost) * len(cost)

    print(f"Ferritin Proxy Data Processed. Shape: {data.shape}")
    print(f"Blocks: {len(block)} (0=Obs, 1=CBC, 2=CMP, 3=Liver)")
    
    return data, block, cost