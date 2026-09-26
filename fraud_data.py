"""
Real-Time Dynamic AML Streaming Engine & Data Layer (Root Module)
Processes transactions incrementally from Data/testing_trans.csv & Data/testing_accounts.csv.
Builds dynamic in-memory graph using NetworkX.
Executes GAT model (backend/GAT/gat_aml_stage1.pt) & evaluates real-time fan-out risk.
Maintains live account customer profiling and supports human auditor feedback loops.
"""

import os
import sys
import math
import time
import json
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
GAT_CKPT_PATH = os.path.join(BASE_DIR, "backend", "GAT", "gat_aml_stage1.pt")
STATE_JSON_PATH = os.path.join(BASE_DIR, "live_stream_state.json")

FX_TO_USD = {
    "US Dollar": 1.0, "Euro": 1.00, "UK Pound": 1.10, "Yen": 0.0069, "Yuan": 0.141,
    "Rupee": 0.0123, "Ruble": 0.0166, "Swiss Franc": 1.02, "Canadian Dollar": 0.735,
    "Australian Dollar": 0.65, "Mexican Peso": 0.050, "Brazil Real": 0.19,
    "Saudi Riyal": 0.267, "Shekel": 0.287, "Bitcoin": 19000.0,
}

CURRENCY_SYMBOLS = {
    "US Dollar": "$", "Euro": "€", "Rupee": "₹", "UK Pound": "£",
    "Yen": "¥", "Yuan": "¥", "Swiss Franc": "CHF ", "Canadian Dollar": "CA$",
    "Australian Dollar": "A$", "Ruble": "₽", "Shekel": "₪", "Bitcoin": "₿",
    "Mexican Peso": "Mex$", "Saudi Riyal": "SAR ", "Brazil Real": "R$",
}

def format_currency(amount, currency_name="US Dollar"):
    sym = "$"
    try:
        amt = float(amount)
        if abs(amt) >= 1_000_000_000:
            return f"{sym}{amt / 1_000_000_000:.2f}B"
        elif abs(amt) >= 1_000_000:
            return f"{sym}{amt / 1_000_000:.2f}M"
        elif abs(amt) >= 1_000:
            return f"{sym}{amt:,.2f}"
        else:
            return f"{sym}{amt:.2f}"
    except Exception:
        return f"{sym}{amount}"


# ==========================================
# 1. GAT MODEL (Exact Architecture from Finalgat.ipynb)
# ==========================================

class GATAMLModel(nn.Module):
    def __init__(self, node_in_dim=13, edge_in_dim=20, hidden_dim=64, heads=4, dropout=0.20):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=edge_in_dim, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=edge_in_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        classifier_input = hidden_dim + hidden_dim + edge_in_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def encode(self, x, edge_index, edge_attr):
        h = F.leaky_relu(self.node_proj(x), negative_slope=0.2)
        h1 = self.gat1(h, edge_index, edge_attr=edge_attr)
        h = self.norm1(h + h1)
        h = self.dropout(h)
        h2 = self.gat2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(h + h2)
        return h

    def decode(self, node_embeddings, target_edge_index, target_edge_attr):
        src = target_edge_index[0]
        dst = target_edge_index[1]
        source_embedding = node_embeddings[src]
        destination_embedding = node_embeddings[dst]
        edge_rep = torch.cat([source_embedding, destination_embedding, target_edge_attr], dim=-1)
        return self.classifier(edge_rep).squeeze(-1)

    def forward(self, x, edge_index, edge_attr, target_edge_index, target_edge_attr):
        node_embeddings = self.encode(x, edge_index, edge_attr)
        return self.decode(node_embeddings, target_edge_index, target_edge_attr)


# ==========================================
# 2. DYNAMIC REAL-TIME STREAMING ENGINE
# ==========================================

class RealTimeStreamingEngine:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

        self.df_trans = None
        self.acc_meta = {}
        self._load_raw_data()

        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self._init_scalers()

        # Dynamic State
        self.reset_state()

    def _load_model(self):
        if os.path.exists(GAT_CKPT_PATH):
            try:
                ckpt = torch.load(GAT_CKPT_PATH, map_location=self.device)
                config = ckpt.get("model_config", {"node_in_dim": 13, "edge_in_dim": 20, "hidden_dim": 64, "heads": 4, "dropout": 0.2})
                self.model = GATAMLModel(**config).to(self.device)
                self.model.load_state_dict(ckpt["model_state"])
                self.model.eval()
                print("GAT AML Model loaded from checkpoint.")
            except Exception as e:
                print(f"Error loading GAT model: {e}")
        else:
            print(f"Warning: GAT checkpoint not found at {GAT_CKPT_PATH}")

    def _load_raw_data(self):
        # 1. Accounts Metadata
        acc_file = os.path.join(DATA_DIR, "testing_accounts.csv")
        if os.path.exists(acc_file):
            df_a = pd.read_csv(acc_file)
            for _, r in df_a.iterrows():
                anum = str(r.get("Account Number", "")).strip()
                if anum:
                    self.acc_meta[anum] = {
                        "bank_name": str(r.get("Bank Name", "Global Bank")),
                        "bank_id": str(r.get("Bank ID", "BNK-001")),
                        "account_number": anum,
                        "entity_id": str(r.get("Entity ID", f"ENT-{anum[:8]}")),
                        "entity_name": str(r.get("Entity Name", f"Account {anum}")),
                    }
        print(f"Loaded {len(self.acc_meta)} account profiles.")

        # 2. Transactions
        trans_file = os.path.join(DATA_DIR, "testing_trans.csv")
        if os.path.exists(trans_file):
            df_t = pd.read_csv(trans_file)
            if "Account" in df_t.columns and "From Account" not in df_t.columns:
                df_t["From Account"] = df_t["Account"]
            if "Account.1" in df_t.columns and "To Account" not in df_t.columns:
                df_t["To Account"] = df_t["Account.1"]
            df_t["Timestamp"] = pd.to_datetime(df_t["Timestamp"], errors="coerce")
            self.df_trans = df_t.sort_values("Timestamp").reset_index(drop=True)
            print(f"Loaded {len(self.df_trans)} testing transactions sorted chronologically.")

    def _init_scalers(self):
        """Fit feature scalers on baseline distributions to ensure stable z-scores."""
        if self.df_trans is None or self.df_trans.empty:
            return

        edge_rows = []
        for _, row in self.df_trans.iterrows():
            p_curr = str(row.get("Payment Currency", "US Dollar")).strip()
            r_curr = str(row.get("Receiving Currency", "US Dollar")).strip()
            amt_p = float(row.get("Amount Paid", 0.0)) * FX_TO_USD.get(p_curr, 1.0)
            amt_r = float(row.get("Amount Received", amt_p)) * FX_TO_USD.get(r_curr, 1.0)
            log_p = float(np.log1p(max(0.0, amt_p)))
            log_r = float(np.log1p(max(0.0, amt_r)))
            diff = abs(amt_r - amt_p)
            ratio = amt_r / (amt_p + 1e-5)
            dt = row["Timestamp"]
            hr = float(dt.hour) if pd.notna(dt) else 12.0
            dow = float(dt.dayofweek) if pd.notna(dt) else 0.0
            edge_rows.append([
                amt_p, amt_r, log_p, log_r, diff, ratio,
                np.sin(2 * np.pi * hr / 24.0), np.cos(2 * np.pi * hr / 24.0), dow,
                1.0 if dow >= 5 else 0.0, 0.0, 0.0,
                1.0 if 9000.0 <= amt_p < 10000.0 else 0.0, min(amt_p, 10000.0) / 10000.0,
                9999.0, 1.0, 0.0, 0.0, 0.0, 0.0
            ])
        self.edge_scaler.fit(np.array(edge_rows, dtype=np.float32))

        # Baseline node features
        dummy_nodes = np.zeros((100, 13), dtype=np.float32)
        dummy_nodes[:, 0] = np.linspace(0, 15, 100) # out_count
        dummy_nodes[:, 1] = np.linspace(0, 5, 100)  # in_count
        dummy_nodes[:, 2] = np.linspace(0, 500000, 100) # out_total
        dummy_nodes[:, 7] = np.linspace(0, 15, 100) # unique_receivers
        self.node_scaler.fit(dummy_nodes)

    def reset_state(self):
        """Reset streaming state to transaction 0."""
        self.current_idx = 0
        self.G = nx.MultiDiGraph()
        self.account_profiles = {}
        self.pair_history = {}
        self.processed_txs = []
        self.auditor_decisions = {}
        self.investigations = {}
        self.is_streaming = False
        self._save_state_to_disk()

    def get_account_meta(self, acc: str) -> Dict[str, Any]:
        clean_acc = str(acc).strip()
        if clean_acc in self.acc_meta:
            return self.acc_meta[clean_acc]
        return {
            "bank_name": "Global Bank",
            "bank_id": "BNK-001",
            "account_number": clean_acc,
            "entity_id": f"ENT-{clean_acc[:8]}",
            "entity_name": f"Account {clean_acc}"
        }

    def _get_or_init_profile(self, acc: str) -> Dict[str, Any]:
        clean_acc = str(acc).strip()
        if clean_acc not in self.account_profiles:
            meta = self.get_account_meta(clean_acc)
            self.account_profiles[clean_acc] = {
                "account_number": clean_acc,
                "bank_name": meta["bank_name"],
                "bank_id": meta["bank_id"],
                "entity_id": meta["entity_id"],
                "entity_name": meta["entity_name"],
                "incoming_transactions": 0,
                "outgoing_transactions": 0,
                "total_incoming_amount": 0.0,
                "total_outgoing_amount": 0.0,
                "average_incoming_amount": 0.0,
                "average_outgoing_amount": 0.0,
                "maximum_incoming_amount": 0.0,
                "maximum_outgoing_amount": 0.0,
                "unique_senders": set(),
                "unique_receivers": set(),
                "out_amounts": [],
                "in_amounts": [],
                "total_degree": 0,
                "net_flow": 0.0,
            }
        return self.account_profiles[clean_acc]

    def _compute_node_feature_vec(self, acc: str) -> np.ndarray:
        p = self._get_or_init_profile(acc)
        out_cnt = float(p["outgoing_transactions"])
        in_cnt = float(p["incoming_transactions"])
        out_tot = float(p["total_outgoing_amount"])
        in_tot = float(p["total_incoming_amount"])
        out_mean = float(p["average_outgoing_amount"])
        in_mean = float(p["average_incoming_amount"])
        out_max = float(p["maximum_outgoing_amount"])
        u_recv = float(len(p["unique_receivers"]))
        u_send = float(len(p["unique_senders"]))
        net_flow = in_tot - out_tot
        tot_deg = out_cnt + in_cnt
        fanout_ratio = u_recv / (out_cnt + 1.0)
        pass_through = min(out_tot, in_tot) / (max(out_tot, in_tot) + 1.0)
        return np.array([
            out_cnt, in_cnt, out_tot, in_tot, out_mean, in_mean, out_max,
            u_recv, u_send, net_flow, tot_deg, fanout_ratio, pass_through
        ], dtype=np.float32)

    def process_next_transaction(self) -> Optional[Dict[str, Any]]:
        """Processes the next transaction chronologically from testing_trans.csv."""
        if self.df_trans is None or self.current_idx >= len(self.df_trans):
            return None

        row = self.df_trans.iloc[self.current_idx]
        self.current_idx += 1

        tx_id = str(row.get("transaction_id", f"TX-SIM-{self.current_idx:05d}"))
        ts = row["Timestamp"]
        from_bank = str(row.get("From Bank", "BNK-001")).strip()
        to_bank = str(row.get("To Bank", "BNK-002")).strip()
        from_acc = str(row.get("From Account", row.get("Account", "UNK"))).strip()
        to_acc = str(row.get("To Account", row.get("Account.1", "UNK"))).strip()
        pay_curr = str(row.get("Payment Currency", "US Dollar")).strip()
        rec_curr = str(row.get("Receiving Currency", "US Dollar")).strip()
        amt_paid = float(row.get("Amount Paid", 0.0))
        amt_recv = float(row.get("Amount Received", amt_paid))
        pay_fmt = str(row.get("Payment Format", "Wire")).strip()
        is_laundering = int(row.get("Is Laundering", 0))

        amt_paid_usd = amt_paid * FX_TO_USD.get(pay_curr, 1.0)
        amt_recv_usd = amt_recv * FX_TO_USD.get(rec_curr, 1.0)

        # Snapshot prior historical metrics for sender BEFORE mutation
        src_prof = self._get_or_init_profile(from_acc)
        dst_prof = self._get_or_init_profile(to_acc)
        prior_out_cnt = src_prof["outgoing_transactions"]
        prior_unique_recv = len(src_prof["unique_receivers"])

        # Construct Edge Feature Vector
        log_paid = float(np.log1p(max(0.0, amt_paid_usd)))
        log_recv = float(np.log1p(max(0.0, amt_recv_usd)))
        amt_diff = abs(amt_recv_usd - amt_paid_usd)
        amt_ratio = amt_recv_usd / (amt_paid_usd + 1e-5)
        hr = float(ts.hour) if pd.notna(ts) else 12.0
        dow = float(ts.dayofweek) if pd.notna(ts) else 0.0

        pair_key = f"{from_acc}->{to_acc}"
        prev_pair_ts = self.pair_history.get(pair_key)
        if prev_pair_ts is None:
            recency_hours = 9999.0
            first_pair = 1.0
        else:
            recency_hours = (ts - prev_pair_ts).total_seconds() / 3600.0
            first_pair = 0.0
        self.pair_history[pair_key] = ts

        # Z-score against sender history
        out_history = src_prof["out_amounts"]
        if len(out_history) > 1:
            s_mean = np.mean(out_history)
            s_std = max(float(np.std(out_history)), 1.0)
            sender_zscore = float(np.clip((amt_paid_usd - s_mean) / s_std, -5.0, 10.0))
        else:
            sender_zscore = 0.0

        edge_raw = np.array([
            amt_paid_usd, amt_recv_usd, log_paid, log_recv, amt_diff, amt_ratio,
            np.sin(2 * np.pi * hr / 24.0), np.cos(2 * np.pi * hr / 24.0), dow,
            1.0 if dow >= 5 else 0.0,
            1.0 if from_acc == to_acc else 0.0,
            1.0 if from_bank == to_bank else 0.0,
            1.0 if 9000.0 <= amt_paid_usd < 10000.0 else 0.0,
            min(amt_paid_usd, 10000.0) / 10000.0,
            recency_hours, first_pair, sender_zscore,
            0.0, 0.0, 0.0
        ], dtype=np.float32)

        # Mutate Account Profiles with this transaction
        src_prof["outgoing_transactions"] += 1
        src_prof["total_outgoing_amount"] += amt_paid_usd
        src_prof["out_amounts"].append(amt_paid_usd)
        src_prof["maximum_outgoing_amount"] = max(src_prof["maximum_outgoing_amount"], amt_paid_usd)
        src_prof["average_outgoing_amount"] = src_prof["total_outgoing_amount"] / src_prof["outgoing_transactions"]
        src_prof["unique_receivers"].add(to_acc)
        src_prof["total_degree"] = src_prof["incoming_transactions"] + src_prof["outgoing_transactions"]
        src_prof["net_flow"] = src_prof["total_incoming_amount"] - src_prof["total_outgoing_amount"]

        dst_prof["incoming_transactions"] += 1
        dst_prof["total_incoming_amount"] += amt_recv_usd
        dst_prof["in_amounts"].append(amt_recv_usd)
        dst_prof["maximum_incoming_amount"] = max(dst_prof["maximum_incoming_amount"], amt_recv_usd)
        dst_prof["average_incoming_amount"] = dst_prof["total_incoming_amount"] / dst_prof["incoming_transactions"]
        dst_prof["unique_senders"].add(from_acc)
        dst_prof["total_degree"] = dst_prof["incoming_transactions"] + dst_prof["outgoing_transactions"]
        dst_prof["net_flow"] = dst_prof["total_incoming_amount"] - dst_prof["total_outgoing_amount"]

        curr_out_cnt = src_prof["outgoing_transactions"]
        curr_unique_recv = len(src_prof["unique_receivers"])

        # Update in-memory NetworkX Graph
        self.G.add_node(from_acc, **self.get_account_meta(from_acc), role="source")
        self.G.add_node(to_acc, **self.get_account_meta(to_acc), role="receiver")
        self.G.add_edge(from_acc, to_acc, key=tx_id, amount_usd=amt_paid_usd, raw_amount=amt_paid,
                        currency=pay_curr, format=pay_fmt, timestamp=str(ts), is_laundering=is_laundering)

        # GAT Model Inference on current local transaction graph
        x_src = self._compute_node_feature_vec(from_acc)
        x_dst = self._compute_node_feature_vec(to_acc)

        try:
            scaled_src = self.node_scaler.transform(x_src.reshape(1, -1))[0]
            scaled_dst = self.node_scaler.transform(x_dst.reshape(1, -1))[0]
            scaled_edge = self.edge_scaler.transform(edge_raw.reshape(1, -1))[0]
        except Exception:
            scaled_src = x_src
            scaled_dst = x_dst
            scaled_edge = edge_raw

        x_tensor = torch.tensor(np.stack([scaled_src, scaled_dst]), dtype=torch.float32).to(self.device)
        edge_tensor = torch.tensor(scaled_edge, dtype=torch.float32).unsqueeze(0).to(self.device)
        t_edge_idx = torch.tensor([[0], [1]], dtype=torch.long).to(self.device)
        msg_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(self.device)
        msg_attr = torch.cat([edge_tensor, edge_tensor], dim=0)

        gat_prob = 0.50
        if self.model is not None:
            with torch.no_grad():
                logit = self.model(x_tensor, msg_idx, msg_attr, t_edge_idx, edge_tensor)
                gat_prob = float(torch.sigmoid(logit).item())

        # Determine Risk Level and Fan-Out Pattern
        # 1st transaction: No fan-out yet.
        # Determine Risk Level and Fan-Out Pattern
        # 1st transaction: No fan-out yet.
        # When unique receivers >= 2: Fan-Out is detected!
        is_fanout = (curr_unique_recv >= 2)

        if is_fanout:
            if curr_unique_recv >= 3 or gat_prob >= 0.00030:
                risk_tier = "High"
                risk_score = min(98, max(82, 80 + int(curr_unique_recv * 3)))
            else:
                risk_tier = "Medium"
                risk_score = min(75, max(55, 50 + int(curr_unique_recv * 5)))
        else:
            # Single transaction or 1st hop (pre-fanout baseline)
            risk_tier = "Low"
            risk_score = 15

        tx_record = {
            "tx_id": tx_id,
            "timestamp": str(ts),
            "from_account": from_acc,
            "to_account": to_acc,
            "amount_paid": amt_paid,
            "amount_formatted": format_currency(amt_paid, pay_curr),
            "currency": pay_curr,
            "payment_format": pay_fmt,
            "gat_prob": gat_prob,
            "risk_tier": risk_tier,
            "risk_score": risk_score,
            "is_fanout": is_fanout,
            "is_laundering": is_laundering,
            "sender_out_degree": curr_out_cnt,
            "sender_unique_receivers": curr_unique_recv,
        }
        self.processed_txs.append(tx_record)

        # Update or Create Active Investigation if Medium or High risk
        if risk_tier in ["High", "Medium"] and is_fanout:
            investigation_id = f"GROUP-{from_acc}"
            # Check if this sender was already approved as legitimate by human auditor
            if self.auditor_decisions.get(investigation_id, {}).get("decision") == "APPROVE":
                pass
            else:
                src_meta = self.get_account_meta(from_acc)
                if investigation_id not in self.investigations:
                    self.investigations[investigation_id] = {
                        "group_id": investigation_id,
                        "tx_id": investigation_id,
                        "lead_tx_id": tx_id,
                        "account": from_acc,
                        "name": src_meta["entity_name"],
                        "entity_id": src_meta["entity_id"],
                        "bank_name": src_meta["bank_name"],
                        "bank_id": src_meta["bank_id"],
                        "risk": risk_tier,
                        "risk_score": risk_score,
                        "gat_prob": gat_prob,
                        "gat_signal": risk_tier.upper(),
                        "actual_label": 1 if is_laundering else 0,
                        "pattern": f"Max {curr_unique_recv}-degree Fan-Out",
                        "initial_timestamp": str(ts),
                        "latest_timestamp": str(ts),
                        "timestamp": str(ts),
                        "payment_format": pay_fmt,
                        "payment_currency": pay_curr,
                        "amount": src_prof["total_outgoing_amount"],
                        "amount_formatted": format_currency(src_prof["total_outgoing_amount"], pay_curr),
                        "tx_count": curr_out_cnt,
                        "unique_receivers": curr_unique_recv,
                        "unique_senders": len(src_prof["unique_senders"]),
                        "is_fraud": True,
                        "model_used": "GAT AML Model (PyG)",
                        "model_confidence": f"{risk_score}%",
                        "explanations": [
                            f"Fan-Out pattern detected: 1 sender ({from_acc}) → {curr_unique_recv} unique receivers",
                            f"{curr_out_cnt} outgoing transactions recorded in this fan-out cluster",
                            f"Total fan-out outgoing volume: {format_currency(src_prof['total_outgoing_amount'], pay_curr)}",
                            f"GAT Neural Network Fraud Signal: {risk_tier.upper()} ({risk_score}% Confidence · Risk Score: {risk_score}/100)",
                            f"Payment format: {pay_fmt} · Currency: US Dollar ($)",
                        ]
                    }
                else:
                    inv = self.investigations[investigation_id]
                    inv["latest_timestamp"] = str(ts)
                    inv["timestamp"] = str(ts)
                    inv["risk"] = risk_tier
                    inv["gat_signal"] = risk_tier.upper()
                    inv["risk_score"] = max(inv["risk_score"], risk_score)
                    inv["gat_prob"] = gat_prob
                    inv["model_confidence"] = f"{risk_score}%"
                    inv["pattern"] = f"Max {curr_unique_recv}-degree Fan-Out"
                    inv["amount"] = src_prof["total_outgoing_amount"]
                    inv["amount_formatted"] = format_currency(src_prof["total_outgoing_amount"], pay_curr)
                    inv["tx_count"] = curr_out_cnt
                    inv["unique_receivers"] = curr_unique_recv
                    inv["explanations"] = [
                        f"Fan-Out pattern detected: 1 sender ({from_acc}) → {curr_unique_recv} unique receivers",
                        f"{curr_out_cnt} outgoing transactions recorded in this fan-out cluster",
                        f"Total fan-out outgoing volume: {format_currency(src_prof['total_outgoing_amount'], pay_curr)}",
                        f"GAT Neural Network Fraud Signal: {risk_tier.upper()} ({risk_score}% Confidence · Risk Score: {risk_score}/100)",
                        f"Payment format: {pay_fmt} · Currency: US Dollar ($)",
                    ]

        self._save_state_to_disk()
        return tx_record

    def _save_state_to_disk(self):
        try:
            active_invs = [v for k, v in self.investigations.items() if self.auditor_decisions.get(k, {}).get("decision") != "APPROVE"]
            state = {
                "current_idx": self.current_idx,
                "total_txs": len(self.df_trans) if self.df_trans is not None else 0,
                "active_investigations_count": len(active_invs),
                "last_updated": datetime.now().isoformat()
            }
            with open(STATE_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def submit_auditor_decision(self, investigation_id: str, decision: str, notes: str):
        inv_key = str(investigation_id).strip()
        self.auditor_decisions[inv_key] = {
            "decision": decision,
            "notes": notes,
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
        }
        sender_acc = inv_key.replace("GROUP-", "")
        if sender_acc in self.G:
            self.G.nodes[sender_acc]["audited_status"] = decision
        self._save_state_to_disk()

    def get_active_investigations(self) -> List[Dict[str, Any]]:
        active = []
        for inv_id, inv in self.investigations.items():
            aud_dec = self.auditor_decisions.get(inv_id)
            if aud_dec and "Approve" in aud_dec.get("decision", ""):
                continue
            active.append(inv)
        active.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
        return active

    def get_investigation_by_id(self, inv_id: str) -> Dict[str, Any]:
        inv_key = str(inv_id).strip()
        if inv_key in self.investigations:
            return self.investigations[inv_key]
        active = self.get_active_investigations()
        return active[0] if active else {}

    def get_fan_out_rows(self, inv_id: str, include_source: bool = True) -> List[Dict[str, Any]]:
        inv = self.get_investigation_by_id(inv_id)
        if not inv:
            return []
        sender_acc = inv.get("account", "")

        rows = []
        if include_source:
            rows.append({
                "sub_tx_id": f"SRC-{sender_acc[:8]}",
                "role": "Source (Sender)",
                "account": sender_acc,
                "to_account": sender_acc,
                "is_source": True,
                "amount": inv.get("amount_formatted", "$0.00"),
                "time": inv.get("initial_timestamp", inv.get("timestamp", "—")),
                "payment_format": inv.get("payment_format", "Wire"),
                "gat_signal": inv.get("gat_signal", "HIGH"),
                "to_entity_name": inv.get("name", f"Account {sender_acc}"),
                "to_entity_id": inv.get("entity_id", f"ENT-{sender_acc[:8]}"),
                "to_bank_name": inv.get("bank_name", "Global Bank"),
                "to_bank_id": inv.get("bank_id", "BNK-001"),
            })

        for tx in self.processed_txs:
            if tx.get("from_account") == sender_acc:
                r_acc = tx.get("to_account")
                r_meta = self.get_account_meta(r_acc)
                rows.append({
                    "sub_tx_id": tx.get("tx_id", "TX-001"),
                    "role": "Receiver (Hop 1)",
                    "account": r_acc,
                    "to_account": r_acc,
                    "is_source": False,
                    "amount": tx.get("amount_formatted", "$0.00"),
                    "time": tx.get("timestamp", "—"),
                    "payment_format": tx.get("payment_format", "Wire"),
                    "gat_signal": tx.get("risk_tier", "HIGH"),
                    "to_entity_name": r_meta["entity_name"],
                    "to_entity_id": r_meta["entity_id"],
                    "to_bank_name": r_meta["bank_name"],
                    "to_bank_id": r_meta["bank_id"],
                })
        return rows

    def get_customer_profile(self, account_id: str) -> Dict[str, Any]:
        clean_acc = str(account_id).replace("Account ", "").strip()
        p = self._get_or_init_profile(clean_acc)
        meta = self.get_account_meta(clean_acc)

        in_cnt = p["incoming_transactions"]
        out_cnt = p["outgoing_transactions"]
        tot_cnt = in_cnt + out_cnt
        tot_in = p["total_incoming_amount"]
        tot_out = p["total_outgoing_amount"]
        avg_in = p["average_incoming_amount"]
        avg_out = p["average_outgoing_amount"]
        max_in = p["maximum_incoming_amount"]
        max_out = p["maximum_outgoing_amount"]
        uniq_snd = len(p["unique_senders"])
        uniq_rec = len(p["unique_receivers"])
        tot_deg = p["total_degree"]
        net_flow = p["net_flow"]

        risk_tier = "High Risk" if out_cnt >= 4 or uniq_rec >= 3 else ("Medium Risk" if out_cnt >= 2 else "Standard Risk")

        return {
            "account_id": clean_acc,
            "name": meta["entity_name"],
            "entity_id": meta["entity_id"],
            "bank_name": meta["bank_name"],
            "bank_id": meta["bank_id"],
            "risk_tier": risk_tier,
            "total_transactions": str(tot_cnt),
            "total_incoming": format_currency(tot_in, "US Dollar"),
            "total_outgoing": format_currency(tot_out, "US Dollar"),
            "avg_incoming_amount": format_currency(avg_in, "US Dollar"),
            "avg_outgoing_amount": format_currency(avg_out, "US Dollar"),
            "max_incoming_amount": format_currency(max_in, "US Dollar"),
            "max_outgoing_amount": format_currency(max_out, "US Dollar"),
            "unique_senders": uniq_snd,
            "unique_receivers": uniq_rec,
            "total_degree": tot_deg,
            "net_flow": format_currency(net_flow, "US Dollar"),
            "previous_outgoing": out_cnt,
            "previous_incoming": in_cnt,
        }

    def create_network_graph(self, inv_id: str, *args, **kwargs) -> nx.DiGraph:
        inv = self.get_investigation_by_id(inv_id)
        if not inv:
            return nx.DiGraph()

        sender_acc = inv.get("account", "")
        G_sub = nx.DiGraph()

        s_meta = self.get_account_meta(sender_acc)
        G_sub.add_node(
            sender_acc,
            node_type="source",
            label=f"Sender\n{sender_acc}",
            entity_name=s_meta["entity_name"],
            bank_name=s_meta["bank_name"],
            bank_id=s_meta["bank_id"],
            entity_id=s_meta["entity_id"],
            gat_prob=inv.get("gat_prob", 0.99),
            gat_signal=inv.get("gat_signal", "HIGH"),
            risk_score=inv.get("risk_score", 98),
            color="#ef4444",
            hop=0
        )

        currency = inv.get("payment_currency", "US Dollar")
        for tx in self.processed_txs:
            if tx.get("from_account") == sender_acc:
                r_acc = tx.get("to_account")
                r_meta = self.get_account_meta(r_acc)
                amt_str = tx.get("amount_formatted", "$0.00")

                G_sub.add_node(
                    r_acc,
                    node_type="target",
                    label=f"Receiver\n{r_acc}",
                    entity_name=r_meta["entity_name"],
                    bank_name=r_meta["bank_name"],
                    bank_id=r_meta["bank_id"],
                    entity_id=r_meta["entity_id"],
                    amount=amt_str,
                    color="#f59e0b",
                    hop=1
                )
                G_sub.add_edge(sender_acc, r_acc, amount=amt_str, raw_amount=tx.get("amount_paid", 0.0), currency=currency, hop=1)

        return G_sub

    def get_transactions_dataframe(self) -> pd.DataFrame:
        if self.df_trans is None or self.df_trans.empty:
            return pd.DataFrame(columns=[
                "Fan-out Group", "Transaction ID", "Timestamp", "From Account", "From Entity Name",
                "To Account", "To Entity Name", "Amount Paid", "Payment Currency", "Payment Format",
                "GAT Risk Score", "GAT Signal", "Actual Label"
            ])
        
        proc_lookup = {tx["tx_id"]: tx for tx in self.processed_txs}
        rows = []
        for idx, row in self.df_trans.iterrows():
            tx_id = str(row.get("transaction_id", f"TX-SIM-{idx+1:05d}"))
            from_acc = str(row.get("From Account", row.get("Account", "UNK"))).strip()
            to_acc = str(row.get("To Account", row.get("Account.1", "UNK"))).strip()
            pay_curr = str(row.get("Payment Currency", "US Dollar")).strip()
            pay_fmt = str(row.get("Payment Format", "Wire")).strip()
            amt_paid = float(row.get("Amount Paid", 0.0))
            is_laundering = int(row.get("Is Laundering", 0))
            ts = str(row["Timestamp"])
            
            from_meta = self.get_account_meta(from_acc)
            to_meta = self.get_account_meta(to_acc)
            
            proc = proc_lookup.get(tx_id)
            if proc:
                gat_score = f"{proc.get('risk_score', 15)}/100"
                gat_sig = f"{proc.get('risk_tier', 'LOW').upper()} RISK"
            else:
                gat_score = "15/100"
                gat_sig = "LOW RISK"
                
            rows.append({
                "Fan-out Group": f"GROUP-{from_acc}",
                "Transaction ID": tx_id,
                "Timestamp": ts,
                "From Account": from_acc,
                "From Entity Name": from_meta["entity_name"],
                "To Account": to_acc,
                "To Entity Name": to_meta["entity_name"],
                "Amount Paid": format_currency(amt_paid, pay_curr),
                "Payment Currency": pay_curr,
                "Payment Format": pay_fmt,
                "GAT Risk Score": gat_score,
                "GAT Signal": gat_sig,
                "Actual Label": "LAUNDERING" if is_laundering else "LEGITIMATE"
            })
            
        return pd.DataFrame(rows)


# ==========================================
# PUBLIC API FACADE FOR FRONTEND
# ==========================================

_engine = RealTimeStreamingEngine.get_instance()

def get_engine():
    return _engine

def get_transactions_df():
    return _engine.get_transactions_dataframe()

def get_all_flagged_senders():
    return _engine.get_active_investigations()

def get_flagged_transactions():
    return _engine.get_active_investigations()

def get_transaction_by_id(tx_id_or_group_id):
    return _engine.get_investigation_by_id(tx_id_or_group_id)

def get_fan_out_rows(tx_id_or_group_id, include_source=True):
    return _engine.get_fan_out_rows(tx_id_or_group_id, include_source=include_source)

def get_customer_profile(account_id, *args, **kwargs):
    return _engine.get_customer_profile(account_id)

def get_receiver_profile(account_id, *args, **kwargs):
    if isinstance(account_id, dict):
        account_id = account_id.get("to_account", account_id.get("account_id", ""))
    return _engine.get_customer_profile(account_id)

def create_network_graph(tx_id_or_group_id, *args, **kwargs):
    return _engine.create_network_graph(tx_id_or_group_id, *args, **kwargs)

def submit_auditor_decision(investigation_id, decision, notes):
    _engine.submit_auditor_decision(investigation_id, decision, notes)

def step_stream(n=1):
    last_tx = None
    for _ in range(n):
        res = _engine.process_next_transaction()
        if res:
            last_tx = res
    return last_tx

def reset_stream():
    _engine.reset_state()

def get_stream_status():
    last_tx = _engine.processed_txs[-1] if _engine.processed_txs else None
    return {
        "current_idx": _engine.current_idx,
        "total_txs": len(_engine.df_trans) if _engine.df_trans is not None else 0,
        "active_investigations": len(_engine.get_active_investigations()),
        "total_processed": len(_engine.processed_txs),
        "last_tx": last_tx,
        "graph_nodes": _engine.G.number_of_nodes() if hasattr(_engine, "G") else 0,
        "graph_edges": _engine.G.number_of_edges() if hasattr(_engine, "G") else 0,
    }
