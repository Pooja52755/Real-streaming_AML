#!/usr/bin/env python3
"""
Overnight GAT Model Retraining Automation Job
==============================================
Executed automatically via GitHub Actions Cron (daily at 02:00 AM IST / 20:30 UTC)
or manually via workflow_dispatch / terminal execution.

Workflow:
1. Pulls human authorizer verdicts from Supabase PostgreSQL (or local mirror fallback).
2. Verifies decision threshold: Retrains ONLY if at least 50 human decisions have been recorded.
3. Fine-tunes the PyTorch Geometric GAT model using binary cross-entropy loss (BCEWithLogitsLoss).
4. Saves retrained weights locally to backend/GAT/gat_aml_retrained.pt.
5. Automatically pushes updated weights to Hugging Face Hub (Pooja52755/gat-aml-fraud-detector)
   so the live deployed Streamlit application seamlessly loads the latest model on refresh.
"""

import os
import sys
import argparse
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add project root directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import database
import fraud_data


def run_job(epochs: int = 8, lr: float = 0.001, min_decisions: int = 50):
    print("=" * 70)
    print("AML FRAUD DETECTION: OVERNIGHT GAT RETRAINING JOB")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: Epochs={epochs}, LR={lr}, Minimum Decisions Threshold={min_decisions}")
    print("=" * 70)

    # 1. Check Database & Decisions Count
    status = database.get_status()
    print(f"[DATABASE] Provider: {status.get('provider')}")
    print(f"[DATABASE] Supabase Connected: {status.get('connected')}, Tables Ready: {status.get('tables_ready')}")

    all_decisions = database.get_auditor_decisions()
    count = len(all_decisions)
    print(f"[DECISIONS] Retrieved {count} human authorizer decisions from Supabase.")

    if count < min_decisions:
        print(f"\n[DECISION THRESHOLD] [SKIP] Only {count} human decisions recorded.")
        print(f"[DECISION THRESHOLD] Minimum requirement is {min_decisions} decisions.")
        print(f"[DECISION THRESHOLD] Retraining skipped. System is awaiting {min_decisions - count} more decisions.")
        print("=" * 70)
        sys.exit(0)

    print(f"\n[DECISION THRESHOLD] [MET] Threshold met ({count} >= {min_decisions}). Starting GAT fine-tuning...")

    # 2. Run Engine Retraining
    engine = fraud_data.RealTimeStreamingEngine.get_instance()
    report = engine.run_overnight_retraining(epochs=epochs, lr=lr, min_decisions=min_decisions)

    # 3. Report Results
    print("\n" + "=" * 70)
    print("RETRAINING SUMMARY & METRICS")
    print("=" * 70)
    print(f"Status           : {report.get('status')}")
    print(f"Audited Samples  : {report.get('total_samples')}")
    print(f"Pre-Loss         : {report.get('pre_loss')}")
    print(f"Post-Loss        : {report.get('post_loss')}")
    print(f"Post-Accuracy    : {report.get('post_accuracy')}%")
    print(f"Post-Precision   : {report.get('post_precision')}%")
    print(f"Post-Recall      : {report.get('post_recall')}%")
    print(f"Local Model Path : {report.get('model_path')}")
    print(f"Hugging Face Hub : {report.get('huggingface_status')}")
    print("=" * 70)

    if report.get("status") == "success":
        print("🎉 Overnight Retraining Job Completed Successfully!")
        sys.exit(0)
    else:
        print(f"⚠️ Retraining ended with notice: {report.get('message')}")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overnight GAT Model Retraining Automation")
    parser.add_argument("--epochs", type=int, default=8, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--min_decisions", type=int, default=50, help="Minimum human decisions required to trigger retraining")
    args = parser.parse_args()

    run_job(epochs=args.epochs, lr=args.lr, min_decisions=args.min_decisions)
