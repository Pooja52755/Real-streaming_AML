"""
AML Real-Time Streaming System Verification Suite
Verifies:
1. Dataset files & integrity (testing_trans.csv, testing_accounts.csv)
2. PyG GAT Model loading and forward pass (gat_aml_stage1.pt)
3. LightGBM Model loading and inference (aml_lightgbm_model.pkl)
4. RealTimeStreamingEngine (fraud_data.py) live streaming, fan-out detection, auditor override, and state persistence
5. FastAPI service connectivity (optional)
"""

import os
import sys
import json
import torch
import requests
import numpy as np
import pandas as pd

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def verify_datasets():
    print_header("1. VERIFYING DATASETS")
    data_path = "Data/testing_data.csv"
    trans_path = "Data/testing_trans.csv"
    acc_path = "Data/testing_accounts.csv"
    
    assert os.path.exists(data_path) or os.path.exists(trans_path), "Missing testing dataset!"
    assert os.path.exists(acc_path), f"Missing {acc_path}"
    
    active_path = data_path if os.path.exists(data_path) else trans_path
    df_t = pd.read_csv(active_path)
    df_a = pd.read_csv(acc_path)
    
    # Strictly verify that testing dataset has NO 'Is Laundering' column
    assert "Is Laundering" not in df_t.columns and "is_laundering" not in df_t.columns, f"CRITICAL ERROR: Laundering column must be absent from {active_path}!"
    print(f" [OK] Confirmed: Laundering column is REMOVED from {active_path} ({len(df_t.columns)} raw feature columns: {list(df_t.columns)}).")
    print(f" [OK] {os.path.basename(active_path)}: {len(df_t)} rows loaded.")
    print(f" [OK] testing_accounts.csv: {len(df_a)} accounts loaded.")
    
    earliest_ts = df_t["Timestamp"].min()
    latest_ts = df_t["Timestamp"].max()
    print(f" [OK] Earliest transaction: {earliest_ts}")
    print(f" [OK] Latest transaction: {latest_ts}")

def verify_models():
    print_header("2. VERIFYING GAT MODEL & DYNAMIC GRAPH INFERENCE")
    
    # GAT
    gat_ckpt = "backend/GAT/gat_aml_stage1.pt"
    assert os.path.exists(gat_ckpt), f"Missing {gat_ckpt}"
    ckpt = torch.load(gat_ckpt, map_location="cpu")
    print(f" [OK] GAT Checkpoint loaded. Total weights keys: {len(ckpt['model_state'])}")
    
    from backend_api import GATAMLModel
    model = GATAMLModel(node_in_dim=13, edge_in_dim=20, hidden_dim=64, heads=4, dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # Test multi-node dynamic graph forward pass (1 sender -> 3 receivers)
    x = torch.randn(4, 13)
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 20)
    target_edge_index = torch.tensor([[0], [3]], dtype=torch.long)
    target_edge_attr = edge_attr[2:3]
    with torch.no_grad():
        logit = model(x, edge_index, edge_attr, target_edge_index, target_edge_attr)
        prob = torch.sigmoid(logit).item()
    print(f" [OK] Dynamic Graph GAT Multi-Head Attention Forward Pass: logit={logit.item():.4f}, prob={prob:.6f}")

def verify_engine():
    print_header("3. VERIFYING REAL-TIME STREAMING ENGINE (fraud_data.py)")
    import fraud_data
    
    fraud_data.reset_stream(clear_decisions=True)
    status_0 = fraud_data.get_stream_status()
    print(f" [OK] Engine Reset: current_idx={status_0['current_idx']}, total_txs={status_0['total_txs']}")
    
    # Step 1
    tx1 = fraud_data.step_stream(1)
    print(f" [OK] Stepped Tx #1: {tx1['tx_id']} | Risk Tier={tx1['risk_tier']} | Score={tx1['risk_score']}/100 | GAT Prob={tx1['gat_prob']:.6f}")
    assert tx1["risk_tier"] != "MODEL_OFFLINE", "GAT model must be active for inference"
    
    # Test that removing model causes MODEL_OFFLINE (proves no hardcoding)
    eng = fraud_data.RealTimeStreamingEngine.get_instance()
    orig_model = eng.model
    eng.model = None
    eng.reset_state(clear_decisions=True)
    tx_offline = eng.process_next_transaction()
    assert tx_offline["risk_tier"] == "MODEL_OFFLINE" and tx_offline["risk_score"] == 0, "Removing GAT model must produce MODEL_OFFLINE"
    print(" [OK] Verified: Removing GAT model triggers MODEL_OFFLINE (no hardcoded fallbacks).")
    eng.model = orig_model
    eng.reset_state(clear_decisions=True)

    # Stream to fan-out (step 12 transactions to reach GROUP-8004943A0 fan-out)
    for _ in range(12):
        fraud_data.step_stream(1)
        
    status_11 = fraud_data.get_stream_status()
    active_invs = fraud_data.get_flagged_transactions()
    print(f" [OK] Stepped to Tx #11: active alerts count = {len(active_invs)}")
    assert len(active_invs) >= 1, "Expected at least 1 active fan-out alert by Tx #11"
    
    lead_inv = active_invs[0]
    print(f" [OK] Detected Fan-Out Group: {lead_inv['group_id']} | Risk={lead_inv['risk']} | Score={lead_inv['risk_score']}/100")
    print(f" [OK] Model Used: {lead_inv['model_used']}")
    
    # Test Auditor Approval Override
    inv_id = lead_inv["group_id"]
    fraud_data.submit_auditor_decision(inv_id, "APPROVE", "Legitimate vendor payments verified by auditor")
    print(f" [OK] Submitted Auditor Approval for {inv_id}")
    
    # Check that approved group is cleared from active investigations
    active_after = fraud_data.get_flagged_transactions()
    assert all(inv["group_id"] != inv_id for inv in active_after), "Approved investigation should be cleared from active alerts"
    print(f" [OK] Approved alert successfully cleared from active investigations.")
    
    # Verify persistence in live_stream_state.json and auditor_decisions.json
    state_file = "live_stream_state.json"
    assert os.path.exists(state_file), f"Missing {state_file}"
    with open(state_file, "r", encoding="utf-8") as f:
        st = json.load(f)
    assert inv_id in st.get("auditor_decisions", {}), "Auditor decision must be persisted in live_stream_state.json"
    print(f" [OK] Auditor decision successfully verified in {state_file}")

    # Verify fanout_decisions.json
    fo_file = "Data/fanout_decisions.json"
    assert os.path.exists(fo_file), f"Missing {fo_file}"
    with open(fo_file, "r", encoding="utf-8") as f:
        fo_data = json.load(f)
    print(f" [OK] Data/fanout_decisions.json verified. Total fan-out episodes recorded: {len(fo_data)}")

def verify_retraining_and_compliance_pipeline():
    print_header("4. VERIFYING OVERNIGHT RETRAINING & MODEL REGISTRY")
    import fraud_data

    # Submit sample transaction decisions for retraining
    proc_txs = fraud_data.get_processed_transactions_list()
    for tx in proc_txs[:5]:
        tx_id = tx["tx_id"]
        dec = "🚫 Confirm Laundering (Block / SAR)" if tx.get("risk_tier") in ["High", "Medium"] else "✅ Approve Transaction (Legitimate)"
        fraud_data.submit_auditor_decision(tx_id, dec, "Verification suite verified case", target_type="transaction")

    # Verify auditor_decisions.json
    aud_file = "Data/auditor_decisions.json"
    assert os.path.exists(aud_file), f"Missing {aud_file}"
    with open(aud_file, "r", encoding="utf-8") as f:
        aud_data = json.load(f)
    assert len(aud_data) >= 5, "Expected at least 5 decisions in auditor_decisions.json"
    print(f" [OK] Data/auditor_decisions.json verified. Total decisions: {len(aud_data)}")

    # Execute Overnight Retraining
    report = fraud_data.run_overnight_retraining(epochs=3, lr=0.001)
    assert report.get("status") == "success", f"Retraining failed: {report}"
    print(f" [OK] Overnight Retraining Succeeded: Pre-Loss={report['pre_loss']}, Post-Loss={report['post_loss']}")
    print(f" [OK] Real Computed Metrics: Post-Accuracy={report['post_accuracy']}%, Precision={report['post_precision']}%, Recall={report['post_recall']}%")

    # Verify model_registry.json
    reg_file = "backend/GAT/model_registry.json"
    assert os.path.exists(reg_file), f"Missing {reg_file}"
    with open(reg_file, "r", encoding="utf-8") as f:
        reg_data = json.load(f)
    assert len(reg_data.get("models", [])) >= 2, "Expected at least 2 models in registry (base + retrained)"
    print(f" [OK] backend/GAT/model_registry.json verified. Active version: {reg_data.get('active_version')}")

    # Verify APScheduler status
    sched = fraud_data.get_scheduler_status()
    print(f" [OK] APScheduler Status: {sched.get('status')} | Schedule: {sched.get('schedule')} | Next Run: {sched.get('next_run')}")

def verify_optional_fastapi():
    print_header("5. CHECKING FASTAPI BACKEND (OPTIONAL)")
    try:
        resp = requests.get("http://localhost:8000/", timeout=1.5)
        if resp.status_code == 200:
            print(" [OK] FastAPI server is running on http://localhost:8000:", resp.json())
        else:
            print(f" [INFO] FastAPI server responded with status: {resp.status_code}")
    except Exception:
        print(" [NOTE] FastAPI server is not currently running on port 8000.")
        print("        (The application is running directly via Streamlit: streamlit run app.py)")

if __name__ == "__main__":
    verify_datasets()
    verify_models()
    verify_engine()
    verify_retraining_and_compliance_pipeline()
    verify_optional_fastapi()
    print("\n" + "=" * 60)
    print(" ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! [OK]")
    print("=" * 60 + "\n")
