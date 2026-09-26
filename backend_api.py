import os
import sys
import time
import math
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backend_api")

FANOUT_MIN_UNIQUE_RECEIVERS = 2
GAT_MEDIUM_THRESHOLD = 0.30
GAT_HIGH_THRESHOLD = 0.70


def detect_fanout(
    sender: str,
    history: List[Dict[str, Any]],
    current_transaction: Dict[str, Any],
) -> Dict[str, Any]:
    """Detect causal fan-out using history strictly before the current transaction."""
    current_timestamp = pd.to_datetime(current_transaction.get("timestamp"), errors="coerce")
    sender_history = [
        item for item in history
        if str(item.get("sender", item.get("account", ""))).strip() == str(sender).strip()
        and (
            pd.isna(current_timestamp)
            or pd.to_datetime(item.get("timestamp"), errors="coerce") < current_timestamp
        )
    ]
    historical_receivers = {
        str(item.get("receiver", "")).strip()
        for item in sender_history
        if str(item.get("receiver", "")).strip()
    }
    current_receiver = str(
        current_transaction.get("receiver", current_transaction.get("receiver_account", ""))
    ).strip()
    receivers_including_current = set(historical_receivers)
    if current_receiver:
        receivers_including_current.add(current_receiver)
    amounts = [float(item.get("amount", item.get("amount_paid", 0.0))) for item in sender_history]
    previous_count = len(sender_history)
    previous_total = sum(amounts)
    previous_unique = len(historical_receivers)
    total_unique = len(receivers_including_current)
    detected = total_unique >= FANOUT_MIN_UNIQUE_RECEIVERS
    candidate = not detected and total_unique >= 2
    return {
        "sender": sender,
        "previous_outgoing_count": previous_count,
        "previous_unique_receivers": previous_unique,
        "previous_outgoing_total": previous_total,
        "previous_outgoing_average": previous_total / previous_count if previous_count else None,
        "previous_outgoing_max": max(amounts) if amounts else None,
        "current_receiver": current_receiver,
        "unique_receivers_including_current": total_unique,
        "fanout_expansion": total_unique - previous_unique,
        "fanout_detected": detected,
        "fanout_candidate": candidate,
        "fanout_status": "ESTABLISHED" if detected else ("CANDIDATE" if candidate else "NOT_ESTABLISHED"),
    }

# ==========================================
# 1. EXACT GAT MODEL ARCHITECTURE (32,385 PARAMS)
# ==========================================

class GATAMLModel(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 13,
        edge_in_dim: int = 20,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.20
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.heads = heads

        self.node_proj = nn.Linear(node_in_dim, hidden_dim)

        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            concat=True,
            edge_dim=edge_in_dim,
            dropout=dropout
        )

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            concat=True,
            edge_dim=edge_in_dim,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        classifier_input = hidden_dim + hidden_dim + edge_in_dim  # 64 + 64 + 20 = 148
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
# 2. FEATURE ENGINE & SCALING
# ==========================================

FX_TO_USD = {
    "US Dollar": 1.0, "USD": 1.0, "$": 1.0,
    "Euro": 1.00, "EUR": 1.00,
    "UK Pound": 1.10, "GBP": 1.10,
    "Yen": 0.0069, "JPY": 0.0069,
    "Yuan": 0.141, "CNY": 0.141,
    "Rupee": 0.0123, "INR": 0.0123,
    "Ruble": 0.0166, "RUB": 0.0166,
    "Swiss Franc": 1.02, "CHF": 1.02,
    "Canadian Dollar": 0.735, "CAD": 0.735,
    "Australian Dollar": 0.65, "AUD": 0.65,
    "Mexican Peso": 0.050, "MXN": 0.050,
    "Brazil Real": 0.19, "BRL": 0.19,
    "Saudi Riyal": 0.267, "SAR": 0.267,
    "Shekel": 0.287, "ILS": 0.287,
    "Bitcoin": 19000.0, "BTC": 19000.0,
}

PAYMENT_FORMAT_MAP = {}
PAYMENT_CURRENCY_MAP = {}
RECEIVING_CURRENCY_MAP = {}

def encode_cat(val: str, mapping: dict) -> float:
    s = str(val).strip()
    if s not in mapping:
        mapping[s] = len(mapping)
    return float(mapping[s])

class GATStreamingEngine:
    def __init__(self, pt_model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(pt_model_path):
            raise FileNotFoundError(f"GAT model checkpoint not found at {pt_model_path}")

        ckpt = torch.load(pt_model_path, map_location=self.device)
        config = ckpt.get("model_config", {"node_in_dim": 13, "edge_in_dim": 20, "hidden_dim": 64, "heads": 4, "dropout": 0.2})
        self.model = GATAMLModel(**config).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Checkpoint: {pt_model_path}")
        logger.info(f"Node dim: {config['node_in_dim']}")
        logger.info(f"Edge dim: {config['edge_in_dim']}")
        logger.info(f"Hidden dim: {config['hidden_dim']}")
        logger.info(f"Heads: {config['heads']}")
        logger.info(f"Parameters: {param_count}")

        if param_count != 32385:
            logger.warning(f"Parameter count mismatch: Expected 32385, got {param_count}")

        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.scaler_initialized = False

        self.node_to_id: Dict[str, int] = {}
        self.id_to_node: Dict[int, str] = {}
        self.edges: List[Dict[str, Any]] = []
        self.account_history: Dict[str, Dict[str, Any]] = {}
        self.auditor_memory: Dict[str, str] = {}
        self.started = False

    def initialize_scalers_from_dataset(self, df: pd.DataFrame):
        df_copy = df.copy()
        df_copy['Timestamp'] = pd.to_datetime(df_copy['Timestamp'], errors='coerce')
        df_copy['Amount Paid'] = pd.to_numeric(df_copy['Amount Paid'], errors='coerce').fillna(0.0)
        df_copy['Amount Received'] = pd.to_numeric(df_copy['Amount Received'], errors='coerce').fillna(0.0)

        paid_fx = df_copy['Payment Currency'].astype(str).str.strip().map(FX_TO_USD).fillna(1.0).to_numpy(dtype=np.float32)
        recv_fx = df_copy['Receiving Currency'].astype(str).str.strip().map(FX_TO_USD).fillna(1.0).to_numpy(dtype=np.float32)
        df_copy['amt_paid_usd'] = df_copy['Amount Paid'].to_numpy(dtype=np.float32) * paid_fx
        df_copy['amt_recv_usd'] = df_copy['Amount Received'].to_numpy(dtype=np.float32) * recv_fx

        df_copy['src_key'] = df_copy['From Bank'].astype(str).str.strip() + '_' + df_copy['Account'].astype(str).str.strip()
        df_copy['dst_key'] = df_copy['To Bank'].astype(str).str.strip() + '_' + df_copy['Account.1'].astype(str).str.strip()

        edge_features = pd.DataFrame(index=df_copy.index)
        edge_features['amount_paid'] = df_copy['amt_paid_usd']
        edge_features['amount_received'] = df_copy['amt_recv_usd']
        edge_features['log_amount_paid'] = np.log1p(df_copy['amt_paid_usd'].clip(lower=0))
        edge_features['log_amount_received'] = np.log1p(df_copy['amt_recv_usd'].clip(lower=0))
        edge_features['amount_difference'] = (df_copy['amt_recv_usd'] - df_copy['amt_paid_usd']).abs()
        edge_features['amount_ratio'] = df_copy['amt_recv_usd'] / (df_copy['amt_paid_usd'] + 1e-5)

        hour = df_copy['Timestamp'].dt.hour.fillna(0).astype(np.float32)
        edge_features['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        edge_features['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        edge_features['day_of_week'] = df_copy['Timestamp'].dt.dayofweek.fillna(0)
        edge_features['weekend_flag'] = (edge_features['day_of_week'] >= 5).astype(np.float32)

        edge_features['self_loop'] = (df_copy['src_key'] == df_copy['dst_key']).astype(np.float32)
        edge_features['same_bank'] = (df_copy['From Bank'].astype(str) == df_copy['To Bank'].astype(str)).astype(np.float32)

        threshold = 10000.0
        edge_features['near_threshold'] = ((df_copy['amt_paid_usd'] >= 9000.0) & (df_copy['amt_paid_usd'] < 10000.0)).astype(np.float32)
        edge_features['threshold_proximity'] = df_copy['amt_paid_usd'].clip(0, threshold) / threshold

        df_copy['pair_key'] = df_copy['src_key'] + '->' + df_copy['dst_key']
        ts_seconds = df_copy['Timestamp'].astype('int64') // 10**9
        previous_pair_time = ts_seconds.groupby(df_copy['pair_key']).shift(1)
        edge_features['pair_recency_hours'] = ((ts_seconds - previous_pair_time) / 3600.0).fillna(9999.0)
        edge_features['first_pair_transaction'] = previous_pair_time.isna().astype(np.float32)

        sender_mean = df_copy.groupby('src_key')['amt_paid_usd'].transform('mean')
        sender_std = df_copy.groupby('src_key')['amt_paid_usd'].transform('std').fillna(1.0).replace(0.0, 1.0)
        edge_features['sender_amount_zscore'] = (df_copy['amt_paid_usd'] - sender_mean) / sender_std

        edge_features['payment_format'] = [encode_cat(x, PAYMENT_FORMAT_MAP) for x in df_copy['Payment Format']]
        edge_features['payment_currency'] = [encode_cat(x, PAYMENT_CURRENCY_MAP) for x in df_copy['Payment Currency']]
        edge_features['receiving_currency'] = [encode_cat(x, RECEIVING_CURRENCY_MAP) for x in df_copy['Receiving Currency']]

        nodes = pd.concat([df_copy['src_key'], df_copy['dst_key']]).unique()
        node_df = pd.DataFrame(index=nodes)
        node_df['out_count'] = df_copy.groupby('src_key').size().reindex(node_df.index).fillna(0)
        node_df['in_count'] = df_copy.groupby('dst_key').size().reindex(node_df.index).fillna(0)
        node_df['out_total'] = df_copy.groupby('src_key')['amt_paid_usd'].sum().reindex(node_df.index).fillna(0)
        node_df['in_total'] = df_copy.groupby('dst_key')['amt_recv_usd'].sum().reindex(node_df.index).fillna(0)
        node_df['out_mean'] = df_copy.groupby('src_key')['amt_paid_usd'].mean().reindex(node_df.index).fillna(0)
        node_df['in_mean'] = df_copy.groupby('dst_key')['amt_recv_usd'].mean().reindex(node_df.index).fillna(0)
        node_df['out_max'] = df_copy.groupby('src_key')['amt_paid_usd'].max().reindex(node_df.index).fillna(0)
        node_df['unique_receivers'] = df_copy.groupby('src_key')['dst_key'].nunique().reindex(node_df.index).fillna(0)
        node_df['unique_senders'] = df_copy.groupby('dst_key')['src_key'].nunique().reindex(node_df.index).fillna(0)
        node_df['net_flow'] = node_df['in_total'] - node_df['out_total']
        node_df['total_degree'] = node_df['out_count'] + node_df['in_count']
        node_df['fanout_ratio'] = node_df['unique_receivers'] / (node_df['out_count'] + 1e-5)
        node_df['pass_through_ratio'] = np.minimum(node_df['out_total'], node_df['in_total']) / (np.maximum(node_df['out_total'], node_df['in_total']) + 1e-5)

        edge_features = edge_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        node_df = node_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.edge_scaler.fit(edge_features)
        self.node_scaler.fit(node_df)
        self.scaler_initialized = True
        logger.info("Initialized StandardScalers from testing dataset successfully.")

    def reset(self):
        self.edges.clear()
        self.node_to_id.clear()
        self.id_to_node.clear()
        self.account_history.clear()
        self.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.edge_attr = torch.empty((0, 20), dtype=torch.float)
        self.auditor_memory.clear()
        self.started = False

    def _get_or_create_node_id(self, node_key: str) -> int:
        if node_key not in self.node_to_id:
            nid = len(self.node_to_id)
            self.node_to_id[node_key] = nid
            self.id_to_node[nid] = node_key
            self.account_history[node_key] = {
                "out_count": 0, "in_count": 0,
                "out_total": 0.0, "in_total": 0.0,
                "out_amounts": [], "unique_receivers": set(),
                "unique_senders": set(), "tx_timestamps": []
            }
        return self.node_to_id[node_key]

    def _compute_raw_node_features(self, node_key: str) -> np.ndarray:
        h = self.account_history.get(node_key, {
            "out_count": 0, "in_count": 0, "out_total": 0.0, "in_total": 0.0,
            "out_amounts": [], "unique_receivers": set(), "unique_senders": set()
        })
        out_cnt = float(h["out_count"])
        in_cnt = float(h["in_count"])
        out_tot = float(h["out_total"])
        in_tot = float(h["in_total"])
        out_mean = (out_tot / out_cnt) if out_cnt > 0 else 0.0
        in_mean = (in_tot / in_cnt) if in_cnt > 0 else 0.0
        out_max = max(h["out_amounts"]) if h["out_amounts"] else 0.0
        uniq_rec = float(len(h["unique_receivers"]))
        uniq_snd = float(len(h["unique_senders"]))
        net_flow = in_tot - out_tot
        total_deg = out_cnt + in_cnt
        fanout_ratio = (uniq_rec / (out_cnt + 1e-5))
        pass_through = min(out_tot, in_tot) / (max(out_tot, in_tot) + 1e-5)

        return np.array([
            out_cnt, in_cnt, out_tot, in_tot, out_mean, in_mean, out_max,
            uniq_rec, uniq_snd, net_flow, total_deg, fanout_ratio, pass_through
        ], dtype=np.float32)

    def _compute_raw_edge_features(self, tx: Dict[str, Any], src_hist: Dict[str, Any]) -> np.ndarray:
        amt_paid = float(tx.get("amount_paid", tx.get("Amount Paid", tx.get("amount_received", 0.0))))
        amt_recv = float(tx.get("amount_received", tx.get("Amount Received", amt_paid)))
        pay_curr = str(tx.get("payment_currency", tx.get("Payment Currency", "USD"))).strip()
        rec_curr = str(tx.get("receiving_currency", tx.get("Receiving Currency", "USD"))).strip()

        fx_paid = FX_TO_USD.get(pay_curr, 1.0)
        fx_recv = FX_TO_USD.get(rec_curr, 1.0)

        usd_paid = amt_paid * fx_paid
        usd_recv = amt_recv * fx_recv

        log_paid = float(np.log1p(max(0.0, usd_paid)))
        log_recv = float(np.log1p(max(0.0, usd_recv)))
        amt_diff = abs(usd_recv - usd_paid)
        amt_ratio = (usd_recv / (usd_paid + 1e-5))

        ts_str = str(tx.get("timestamp", tx.get("Timestamp", "")))
        try:
            dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
        except Exception:
            try:
                dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M")
            except Exception:
                try:
                    dt = datetime.fromisoformat(ts_str)
                except Exception:
                    dt = datetime.now()

        hour = float(dt.hour)
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        day_of_week = float(dt.weekday())
        weekend_flag = 1.0 if day_of_week >= 5 else 0.0

        self_loop = 1.0 if str(tx.get("from_bank")) == str(tx.get("to_bank")) and str(tx.get("account")) == str(tx.get("receiver_account")) else 0.0
        same_bank = 1.0 if str(tx.get("from_bank")) == str(tx.get("to_bank")) else 0.0

        near_threshold = 1.0 if 9000.0 <= usd_paid < 10000.0 else 0.0
        threshold_proximity = min(usd_paid, 10000.0) / 10000.0

        pair_recency = 9999.0
        first_pair = 1.0
        out_amounts = src_hist.get("out_amounts", [])
        if out_amounts:
            mean_amt = np.mean(out_amounts)
            std_amt = max(float(np.std(out_amounts)), 1.0)
            raw_zscore = float((usd_paid - mean_amt) / std_amt)
            sender_zscore = float(np.clip(raw_zscore, -5.0, 10.0))
        else:
            sender_zscore = 0.0

        fmt_code = encode_cat(tx.get("payment_format", tx.get("Payment Format", "ACH")), PAYMENT_FORMAT_MAP)
        pay_curr_code = encode_cat(pay_curr, PAYMENT_CURRENCY_MAP)
        rec_curr_code = encode_cat(rec_curr, RECEIVING_CURRENCY_MAP)

        return np.array([
            usd_paid, usd_recv, log_paid, log_recv, amt_diff, amt_ratio,
            hour_sin, hour_cos, day_of_week, weekend_flag, self_loop, same_bank,
            near_threshold, threshold_proximity, pair_recency, first_pair,
            sender_zscore, fmt_code, pay_curr_code, rec_curr_code
        ], dtype=np.float32)

    def process_transaction(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        # NO ground truth Is Laundering column used anywhere for prediction!
        from_bank = str(tx.get("from_bank", tx.get("From Bank", "0"))).strip()
        from_acc = str(tx.get("account", tx.get("Account", tx.get("From Account", "UNK")))).strip()
        to_bank = str(tx.get("to_bank", tx.get("To Bank", "0"))).strip()
        to_acc = str(tx.get("receiver_account", tx.get("Account.1", tx.get("To Account", "UNK")))).strip()

        src_key = f"{from_bank}_{from_acc}"
        dst_key = f"{to_bank}_{to_acc}"

        # Snapshot the graph before creating current-transaction nodes.
        prior_graph_edges = len(self.edges)
        prior_graph_nodes = len(self.node_to_id)

        # 1. Build features using prior state BEFORE updating history (No target data leakage!)
        src_id = self._get_or_create_node_id(src_key)
        dst_id = self._get_or_create_node_id(dst_key)
        src_hist = self.account_history[src_key]
        dst_hist = self.account_history[dst_key]

        # SNAPSHOT CAUSAL PRIOR METRICS BEFORE MUTATION
        prior_src_out_cnt = int(src_hist["out_count"])
        prior_src_in_cnt = int(src_hist["in_count"])
        prior_src_out_tot = float(src_hist["out_total"])
        prior_src_in_tot = float(src_hist["in_total"])
        prior_src_uniq_recs = len(src_hist["unique_receivers"])
        prior_src_uniq_snds = len(src_hist["unique_senders"])
        prior_src_out_mean = (prior_src_out_tot / prior_src_out_cnt) if prior_src_out_cnt > 0 else 0.0
        prior_src_out_max = max(src_hist["out_amounts"]) if src_hist["out_amounts"] else 0.0

        prior_dst_out_cnt = int(dst_hist["out_count"])
        prior_dst_in_cnt = int(dst_hist["in_count"])
        prior_dst_out_tot = float(dst_hist["out_total"])
        prior_dst_in_tot = float(dst_hist["in_total"])
        prior_dst_uniq_recs = len(dst_hist["unique_receivers"])
        prior_dst_uniq_snds = len(dst_hist["unique_senders"])
        prior_dst_out_mean = (prior_dst_out_tot / prior_dst_out_cnt) if prior_dst_out_cnt > 0 else 0.0
        prior_dst_out_max = max(dst_hist["out_amounts"]) if dst_hist["out_amounts"] else 0.0

        raw_x_src = self._compute_raw_node_features(src_key)
        raw_x_dst = self._compute_raw_node_features(dst_key)
        raw_edge_feat = self._compute_raw_edge_features(tx, src_hist)

        if self.scaler_initialized:
            scaled_src = self.node_scaler.transform(raw_x_src.reshape(1, -1))[0]
            scaled_dst = self.node_scaler.transform(raw_x_dst.reshape(1, -1))[0]
            scaled_edge = self.edge_scaler.transform(raw_edge_feat.reshape(1, -1))[0]
        else:
            scaled_src = np.sign(raw_x_src) * np.log1p(np.abs(raw_x_src))
            scaled_dst = np.sign(raw_x_dst) * np.log1p(np.abs(raw_x_dst))
            scaled_edge = raw_edge_feat

        x_nodes = torch.tensor(np.stack([scaled_src, scaled_dst]), dtype=torch.float32).to(self.device)
        edge_attr = torch.tensor(scaled_edge, dtype=torch.float32).unsqueeze(0).to(self.device)

        target_edge_index = torch.tensor([[0], [1]], dtype=torch.long).to(self.device)
        msg_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(self.device)
        msg_edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        # 2. PURE GAT MODEL FORWARD PASS (Logit -> Sigmoid)
        with torch.no_grad():
            logit = self.model(x_nodes, msg_edge_index, msg_edge_attr, target_edge_index, edge_attr)
            raw_logit_val = float(logit.item() if logit.numel() == 1 else logit[0].item())
            gat_probability = float(torch.sigmoid(torch.tensor(raw_logit_val)).item())

        # 3. Calculate Risk Score strictly from GAT Probability
        derived_risk_score = int(round(gat_probability * 100))

        amt_paid = float(tx.get("amount_paid", tx.get("Amount Paid", tx.get("amount_received", 0.0))))
        tx_id = str(tx.get("transaction_id", f"TX-{len(self.edges)+1:05d}"))
        ts_str = str(tx.get("timestamp", tx.get("Timestamp", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))))

        # Fan-out is evaluated from the pre-current transaction history only.
        fanout_result = detect_fanout(
            from_acc,
            self.edges,
            {"receiver": to_acc, "amount": amt_paid, "timestamp": ts_str},
        )
        fanout_detected = bool(fanout_result["fanout_detected"])
        fanout_candidate = bool(fanout_result["fanout_candidate"])
        pattern = "FAN-OUT" if fanout_detected else ("FAN-OUT CANDIDATE" if fanout_candidate else "NONE")

        # GAT model risk remains independent from the operational fan-out alert.
        if gat_probability >= GAT_HIGH_THRESHOLD:
            risk_level = "HIGH"
        elif gat_probability >= GAT_MEDIUM_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        fanout_alert = fanout_detected and risk_level in {"HIGH", "MEDIUM"}
        fanout_alert_level = f"{risk_level}_FANOUT" if fanout_alert else (
            "FANOUT_CANDIDATE" if fanout_detected else "NONE"
        )

        # 5. DETERMINISTIC & CAUSAL EXPLANATION ENGINE
        is_first_tx = (prior_src_out_cnt == 0 and prior_src_in_cnt == 0 and prior_graph_edges == 0)

        behavioral_signals = []
        if not fanout_detected:
            behavioral_signals = [
                f"Previous transactions: <b>0</b>",
                f"Previous outgoing transactions: <b>0</b>",
                f"Previous receivers: <b>0</b>",
                f"Historical graph connections: <b>0</b>"
            ]
            first_tx_interpretation = (
                "This is the first observed transaction for this sender in the current stream. "
                "There is no historical behavioral pattern available yet. The GAT produced a high-risk signal, "
                "but the dashboard should not attribute the signal to historical fan-out or repeated behavior."
            )
            summary_message = "The GAT model signal did not establish a causal fan-out alert."
        else:
            first_tx_interpretation = None
            summary_message = None
            behavioral_signals.append(f"Sender had <b>{prior_src_out_cnt} previous outgoing transactions</b>")
            behavioral_signals.append(f"Sender had <b>{prior_src_uniq_recs} unique historical receivers</b>")
            if prior_src_out_mean > 0 and amt_paid > prior_src_out_mean * 1.2:
                ratio = amt_paid / prior_src_out_mean
                behavioral_signals.append(f"Current amount is <b>{ratio:.1f} times</b> the historical average (${prior_src_out_mean:,.2f})")
            else:
                behavioral_signals.append(f"Historical average outgoing amount: <b>${prior_src_out_mean:,.2f}</b>")

            behavioral_signals.append(
                f"Unique receivers including current transaction: <b>{fanout_result['unique_receivers_including_current']}</b>"
            )

        account_context = [
            "KYC: <b>Verified</b>",
            "Account type: <b>Checking</b>",
            "Location: <b>New York, USA</b>"
        ]

        explanation_payload = {
            "is_first_tx": is_first_tx,
            "fanout": fanout_result,
            "model_signal": {
                "gat_risk_probability": f"{gat_probability * 100:.4f}%",
                "gat_risk_probability_formatted": f"{gat_probability * 100:.2f}%",
                "raw_probability": gat_probability,
                "raw_logit": raw_logit_val
            },
            "behavioral_signals": behavioral_signals,
            "account_context": account_context,
            "first_tx_interpretation": first_tx_interpretation,
            "summary_message": summary_message,
            "investigator_note": "These signals are model/investigation indicators and do not constitute a confirmed fraud decision."
        }

        # 6. CAUSAL CUSTOMER PROFILES (STRICTLY BEFORE CURRENT TX)
        sender_profile = {
            "account_id": from_acc,
            "bank_id": from_bank,
            "historical_out_count": prior_src_out_cnt,
            "historical_in_count": prior_src_in_cnt,
            "historical_out_total": prior_src_out_tot,
            "historical_in_total": prior_src_in_tot,
            "historical_out_mean": prior_src_out_mean,
            "historical_out_max": prior_src_out_max,
            "unique_receivers_count": prior_src_uniq_recs,
            "unique_senders_count": prior_src_uniq_snds
        }

        receiver_profile = {
            "account_id": to_acc,
            "bank_id": to_bank,
            "historical_out_count": prior_dst_out_cnt,
            "historical_in_count": prior_dst_in_cnt,
            "historical_out_total": prior_dst_out_tot,
            "historical_in_total": prior_dst_in_tot,
            "historical_out_mean": prior_dst_out_mean,
            "historical_out_max": prior_dst_out_max,
            "unique_receivers_count": prior_dst_uniq_recs,
            "unique_senders_count": prior_dst_uniq_snds
        }

        debug_features = {
            "sender_account": from_acc,
            "receiver_account": to_acc,
            "sender_raw_node_features": raw_x_src.tolist(),
            "receiver_raw_node_features": raw_x_dst.tolist(),
            "raw_edge_features": raw_edge_feat.tolist(),
            "historical_node_count": prior_graph_nodes,
            "historical_edge_count": prior_graph_edges,
            "raw_gat_logit": raw_logit_val,
            "sigmoid_probability": gat_probability
        }

        result = {
            "transaction_id": tx_id,
            "risk_probability": round(gat_probability, 6),
            "gat_probability": f"{gat_probability * 100:.2f}%",
            "raw_gat_prob": gat_probability,
            "risk_score": derived_risk_score,
            "derived_risk_score": derived_risk_score,
            "risk_level": risk_level,
            "gat_risk_level": risk_level,
            "fanout_detected": fanout_detected,
            "fanout_candidate": fanout_candidate,
            "fanout": fanout_result,
            "fanout_status": fanout_result["fanout_status"],
            "fanout_alert": fanout_alert,
            "alert": fanout_alert,
            "alert_level": fanout_alert_level,
            "pattern": pattern,
            "sender": from_acc,
            "sender_bank": from_bank,
            "receiver": to_acc,
            "receiver_bank": to_bank,
            "amount": amt_paid,
            "timestamp": ts_str,
            "payment_format": str(tx.get("payment_format", tx.get("Payment Format", "ACH"))),
            "payment_currency": str(tx.get("payment_currency", tx.get("Payment Currency", "USD"))),
            "receiving_currency": str(tx.get("receiving_currency", tx.get("Receiving Currency", "USD"))),
            "investigation_status": "PENDING REVIEW" if fanout_alert else "AUTO CLEARED",
            "human_decision": "Not yet reviewed",
            "dashboard_risk_level": risk_level,
            "graph_state": {
                "total_nodes_in_graph": prior_graph_nodes,
                "total_edges_in_graph": prior_graph_edges,
                "sender_prior_outgoing_count": prior_src_out_cnt,
                "sender_prior_unique_receivers": prior_src_uniq_recs
            },
            "sender_profile": sender_profile,
            "receiver_profile": receiver_profile,
            "explanation": explanation_payload,
            "debug_features": debug_features,
            "recommendation": "Review causal fan-out evidence and customer graph topology." if fanout_alert else "No fan-out alert. Retain GAT signal for model monitoring."
        }

        # 7. AFTER prediction and snapshot, update historical graph state for future transactions
        src_hist["out_count"] += 1
        src_hist["out_total"] += amt_paid
        src_hist["out_amounts"].append(amt_paid)
        src_hist["unique_receivers"].add(dst_key)
        src_hist["tx_timestamps"].append(result["timestamp"])

        dst_hist["in_count"] += 1
        dst_hist["in_total"] += amt_paid
        dst_hist["unique_senders"].add(src_key)

        self.edges.append(result)
        self.started = True
        return result


# ==========================================
# 3. FASTAPI SERVER LIFECYCLE & ROUTES
# ==========================================

app = FastAPI(title="AML GAT Streaming Backend API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[GATStreamingEngine] = None
investigation_history_log: List[Dict[str, Any]] = []
available_transactions = 0
account_metadata: Dict[str, Dict[str, Any]] = {}

def init_engine():
    global engine, available_transactions, account_metadata
    if engine is None:
        pt_path = os.path.join(os.path.dirname(__file__), "backend", "GAT", "gat_aml_stage1.pt")
        if not os.path.exists(pt_path):
            pt_path = "backend/GAT/gat_aml_stage1.pt"
        logger.info(f"Initializing GAT Engine with checkpoint: {pt_path}")
        engine = GATStreamingEngine(pt_path)

        csv_path = "Data/testing_accounts.csv"
        if not os.path.exists(csv_path):
            for cand in ["Data/testing_accounts.csv", "Data/HI-Small_FANOUT_testing_data.csv"]:
                if os.path.exists(cand):
                    csv_path = cand
                    break
        if os.path.exists(csv_path):
            df_init = pd.read_csv(csv_path)
            available_transactions = int(len(df_init))
            if 'Timestamp' in df_init.columns:
                engine.initialize_scalers_from_dataset(df_init)
        account_metadata = {}
        for metadata_path in [
            "Data/testing_trans.csv",
            "Data/HI-Small_FANOUT_10M_accounts.csv",
            "Data/HI-Small_FANOUT_testing_accounts.csv",
        ]:
            if os.path.exists(metadata_path):
                try:
                    metadata_df = pd.read_csv(metadata_path)
                    if "Account Number" in metadata_df.columns:
                        m_clean = metadata_df.dropna(subset=["Account Number"]).copy()
                        m_clean["Account Number"] = m_clean["Account Number"].astype(str).str.strip()

                        acc_nums = m_clean["Account Number"].values
                        bnames = m_clean.get("Bank Name", pd.Series("Global Trust Bank", index=m_clean.index)).fillna("Global Trust Bank").astype(str).values
                        bids = m_clean.get("Bank ID", pd.Series("BNK-001", index=m_clean.index)).fillna("BNK-001").astype(str).values
                        eids = m_clean.get("Entity ID", pd.Series("ENT-000", index=m_clean.index)).fillna("ENT-000").astype(str).values
                        enames = m_clean.get("Entity Name", pd.Series("Account User", index=m_clean.index)).fillna("Account User").astype(str).values
                        atypes = m_clean.get("Account Type", pd.Series("Standard Checking", index=m_clean.index)).fillna("Standard Checking").astype(str).values
                        kycs = m_clean.get("KYC Status", pd.Series("Verified KYC", index=m_clean.index)).fillna("Verified KYC").astype(str).values
                        cities = m_clean.get("City", pd.Series("New York, USA", index=m_clean.index)).fillna("New York, USA").astype(str).values
                        odates = m_clean.get("Open Date", pd.Series("2021-04-12", index=m_clean.index)).fillna("2021-04-12").astype(str).values

                        for anum, bname, bid, eid, ename, atype, kyc, city, odate in zip(
                            acc_nums, bnames, bids, eids, enames, atypes, kycs, cities, odates
                        ):
                            if anum not in account_metadata:
                                account_metadata[anum] = {
                                    "bank_name": bname,
                                    "bank_id": bid,
                                    "entity_id": eid,
                                    "entity_name": ename,
                                    "account_type": atype,
                                    "kyc_status": kyc,
                                    "location": city,
                                    "open_date": odate,
                                }
                except Exception as e:
                    logger.warning(f"Error loading metadata from {metadata_path}: {e}")

# Initialize immediately on import
init_engine()

@app.on_event("startup")
def startup_event():
    init_engine()

@app.get("/")
def read_root():
    return {
        "status": "LIVE" if engine and engine.started else "WAITING",
        "service": "AML GAT Fraud Detection Backend",
        "processed_transactions": len(engine.edges) if engine else 0,
        "available_transactions": available_transactions
    }


def _get_processed_transaction(tx_id: str) -> Dict[str, Any]:
    if not engine:
        raise HTTPException(status_code=500, detail="GAT Engine not initialized")
    for item in engine.edges:
        if item["transaction_id"] == tx_id:
            return item
    raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")


def _timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def _causal_history(tx_id: str) -> List[Dict[str, Any]]:
    current = _get_processed_transaction(tx_id)
    current_ts = _timestamp(current.get("timestamp"))
    if pd.isna(current_ts):
        return [item for item in engine.edges if item["transaction_id"] != tx_id]
    return [
        item for item in engine.edges
        if item["transaction_id"] != tx_id
        and _timestamp(item.get("timestamp")) < current_ts
    ]


def _profile(account_id: str, tx_id: str, incoming: bool = False) -> Dict[str, Any]:
    current = _get_processed_transaction(tx_id)
    history = _causal_history(tx_id)
    account = str(account_id).strip()
    outgoing = [item for item in history if str(item.get("sender", "")).strip() == account]
    incoming_items = [item for item in history if str(item.get("receiver", "")).strip() == account]
    amounts_out = [float(item.get("amount", 0.0)) for item in outgoing]
    amounts_in = [float(item.get("amount", 0.0)) for item in incoming_items]
    selected = incoming_items if incoming else outgoing
    bank = current.get("receiver_bank" if incoming else "sender_bank", "0")
    metadata = dict(account_metadata.get(account, {}))
    current_ts = _timestamp(current.get("timestamp"))
    open_ts = _timestamp(metadata.get("open_date"))
    metadata["account_age_days"] = (
        max(0, int((current_ts - open_ts).days))
        if not pd.isna(current_ts) and not pd.isna(open_ts) else None
    )
    return {
        "account_id": account,
        "bank_id": bank,
        "historical_incoming_transactions": len(incoming_items),
        "historical_unique_senders": len({item.get("sender") for item in incoming_items}),
        "historical_incoming_amount": sum(amounts_in),
        "historical_outgoing_transactions": len(outgoing),
        "historical_unique_receivers": len({item.get("receiver") for item in outgoing}),
        "historical_outgoing_amount": sum(amounts_out),
        "historical_average_outgoing_amount": (sum(amounts_out) / len(amounts_out) if amounts_out else None),
        "historical_maximum_outgoing_amount": (max(amounts_out) if amounts_out else None),
        "historical_minimum_outgoing_amount": (min(amounts_out) if amounts_out else None),
        "current_transaction_amount": float(current.get("amount", 0.0)),
        "current_transaction_id": current["transaction_id"],
        "current_transaction_timestamp": current.get("timestamp"),
        "account_metadata": {"account_id": account, "bank_id": bank, **metadata},
        "historical_transactions": selected,
    }


def _explanation(tx_id: str) -> Dict[str, Any]:
    current = _get_processed_transaction(tx_id)
    history = _causal_history(tx_id)
    sender = str(current.get("sender", ""))
    sender_history = [item for item in history if str(item.get("sender", "")) == sender]
    amounts = [float(item.get("amount", 0.0)) for item in sender_history]
    probability = float(current.get("risk_probability", current.get("raw_gat_prob", 0.0)))
    average = sum(amounts) / len(amounts) if amounts else None
    maximum = max(amounts) if amounts else None
    unique_receivers = len({item.get("receiver") for item in sender_history})
    timestamp = _timestamp(current.get("timestamp"))
    pattern = current.get("pattern", "NONE")
    sender_metadata = account_metadata.get(sender, {})
    return {
        "model_signal": {
            "raw_logit": current.get("debug_features", {}).get("raw_gat_logit"),
            "gat_risk_probability": probability,
            "derived_risk_score": int(round(probability * 100)),
            "model_risk": current.get("risk_level"),
        },
        "historical_behavior": {
            "previous_outgoing_transactions": len(sender_history),
            "previous_unique_receivers": unique_receivers,
            "historical_outgoing_amount": sum(amounts),
            "historical_average_outgoing_amount": average,
            "historical_maximum_outgoing_amount": maximum,
            "current_transaction_amount": float(current.get("amount", 0.0)),
            "amount_vs_average": (float(current.get("amount", 0.0)) / average if average else None),
            "amount_vs_maximum": (float(current.get("amount", 0.0)) / maximum if maximum else None),
        },
        "pattern_signal": {
            "pattern": pattern,
            "fanout_detected": current.get("fanout_detected", False),
            "previous_unique_receivers": unique_receivers,
            "unique_receivers_including_current": current.get("fanout", {}).get(
                "unique_receivers_including_current", unique_receivers
            ),
        },
        "account_context": {
            "sender": sender,
            "kyc": sender_metadata.get("kyc_status"),
            "account_type": sender_metadata.get("account_type"),
            "location": sender_metadata.get("location"),
            "open_date": sender_metadata.get("open_date"),
        },
        "transaction_context": {
            "amount": current.get("amount"),
            "timestamp": current.get("timestamp"),
            "hour": (int(timestamp.hour) if not pd.isna(timestamp) else None),
            "day": (timestamp.day_name() if not pd.isna(timestamp) else None),
            "weekend": (bool(timestamp.weekday() >= 5) if not pd.isna(timestamp) else None),
            "payment_format": current.get("payment_format"),
            "payment_currency": current.get("payment_currency"),
            "receiving_currency": current.get("receiving_currency"),
        },
        "investigator_note": (
            f"The GAT model generated a {probability * 100:.4f}% risk probability using "
            "the information available at scoring time. These signals support investigation "
            "and do not represent a confirmed fraud decision."
        ),
        "causal_history_count": len(history),
    }

@app.post("/api/transactions/stream")
def stream_transaction(payload: Dict[str, Any] = Body(...)):
    if not engine:
        raise HTTPException(status_code=500, detail="GAT Engine not initialized")
    try:
        result = engine.process_transaction(payload)
        return result
    except Exception as e:
        logger.error(f"Error processing transaction stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stream/status")
def get_stream_status():
    processed = len(engine.edges) if engine else 0
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    fanout_candidates = 0
    fanout_alerts = 0
    high_fanout = 0
    medium_fanout = 0
    if engine:
        for item in engine.edges:
            counts[str(item.get("risk_level", "LOW")).upper()] = counts.get(
                str(item.get("risk_level", "LOW")).upper(), 0
            ) + 1
            if item.get("fanout_candidate"):
                fanout_candidates += 1
            if item.get("fanout_alert"):
                fanout_alerts += 1
                if item.get("risk_level") == "HIGH":
                    high_fanout += 1
                elif item.get("risk_level") == "MEDIUM":
                    medium_fanout += 1
    return {
        "started": bool(engine and engine.started),
        "processed_count": processed,
        "high_count": counts["HIGH"],
        "medium_count": counts["MEDIUM"],
        "low_count": counts["LOW"],
        "fanout_candidates": fanout_candidates,
        "fanout_alerts": fanout_alerts,
        "high_fanout": high_fanout,
        "medium_fanout": medium_fanout,
        "available_transactions": available_transactions,
        "stream_status": "LIVE STREAM ACTIVE" if processed else "WAITING FOR LIVE STREAM",
    }


@app.post("/api/stream/reset")
def reset_stream():
    return reset_system_state()

@app.get("/api/transactions")
def get_all_transactions():
    if not engine:
        return []
    return engine.edges

@app.post("/api/system/reset")
def reset_system_state():
    if not engine:
        raise HTTPException(status_code=500, detail="GAT Engine not initialized")
    engine.reset()
    investigation_history_log.clear()
    logger.info("System state, graph memory, and transactions reset to 0.")
    return {"status": "success", "message": "Graph memory and transactions reset to 0"}

@app.get("/api/transactions/{tx_id}")
def get_transaction_by_id(tx_id: str):
    return _get_processed_transaction(tx_id)


@app.get("/api/transactions/{tx_id}/graph")
def get_transaction_graph(tx_id: str):
    current = _get_processed_transaction(tx_id)
    history = _causal_history(tx_id)
    sender = str(current.get("sender", "")).strip()

    # Filter to transactions from this sender (causal fan-out branches)
    sender_history = [
        item for item in history 
        if str(item.get("sender", "")).strip() == sender
    ]

    edges = [
        {
            "transaction_id": item["transaction_id"],
            "sender": item["sender"],
            "receiver": item["receiver"],
            "amount": item["amount"],
            "timestamp": item["timestamp"],
            "payment_format": item.get("payment_format"),
            "status": "historical",
            "risk_probability": item.get("risk_probability"),
        }
        for item in sender_history
    ]
    edges.append({
        "transaction_id": current["transaction_id"],
        "sender": current["sender"],
        "receiver": current["receiver"],
        "amount": current["amount"],
        "timestamp": current["timestamp"],
        "payment_format": current.get("payment_format"),
        "status": "current",
        "risk_probability": current.get("risk_probability"),
    })
    node_ids = sorted({edge["sender"] for edge in edges} | {edge["receiver"] for edge in edges})
    return {
        "transaction_id": tx_id,
        "nodes": [{"id": node_id, "label": node_id} for node_id in node_ids],
        "edges": edges,
        "historical_edge_count": len(sender_history),
        "current_edge_count": 1,
    }


@app.get("/api/transactions/{tx_id}/profile")
def get_transaction_profile(tx_id: str):
    current = _get_processed_transaction(tx_id)
    return {
        "transaction_id": tx_id,
        "sender": _profile(current["sender"], tx_id),
        "receiver": _profile(current["receiver"], tx_id, incoming=True),
    }


@app.get("/api/transactions/{tx_id}/explanation")
def get_transaction_explanation(tx_id: str):
    return _explanation(tx_id)

@app.get("/api/alerts")
def get_active_alerts():
    if not engine:
        return []
    alerts = []
    for e in engine.edges:
        if e.get("fanout_alert") and e.get("investigation_status") == "PENDING REVIEW":
            alerts.append(e)
    return alerts[::-1]

@app.get("/api/customers/{account_id}")
def get_customer_profile_api(account_id: str):
    if not engine:
        raise HTTPException(status_code=500, detail="GAT Engine not initialized")
    
    clean_id = str(account_id).strip()
    
    in_cnt = 0
    in_tot = 0.0
    out_cnt = 0
    out_tot = 0.0
    senders = set()
    receivers = set()
    first_seen = "N/A"
    last_seen = "N/A"
    bank_id = "0"
    
    for e in engine.edges:
        sender_acc = str(e.get("sender", "")).strip()
        recv_acc = str(e.get("receiver", "")).strip()
        amt = float(e.get("amount", 0.0))
        ts = str(e.get("timestamp", ""))
        
        if sender_acc == clean_id or sender_acc.endswith(clean_id):
            out_cnt += 1
            out_tot += amt
            receivers.add(recv_acc)
            bank_id = e.get("sender_bank", bank_id)
            if first_seen == "N/A":
                first_seen = ts
            last_seen = ts
            
        if recv_acc == clean_id or recv_acc.endswith(clean_id):
            in_cnt += 1
            in_tot += amt
            senders.add(sender_acc)
            bank_id = e.get("receiver_bank", bank_id)
            if first_seen == "N/A":
                first_seen = ts
            last_seen = ts

    return {
        "account_id": clean_id,
        "bank_id": bank_id,
        "historical_out_count": out_cnt,
        "historical_in_count": in_cnt,
        "historical_out_total": out_tot,
        "historical_in_total": in_tot,
        "unique_senders_count": max(len(senders), 1 if in_cnt > 0 else 0),
        "unique_receivers_count": max(len(receivers), 1 if out_cnt > 0 else 0),
        "first_seen": first_seen,
        "last_seen": last_seen,
        "entity_name": f"Account {clean_id}"
    }

@app.get("/api/graph")
def get_graph_data():
    if not engine:
        return {"nodes": [], "edges": []}
    nodes = [{"id": nid, "label": nkey} for nkey, nid in engine.node_to_id.items()]
    edges = []
    for e in engine.edges:
        edges.append({
            "tx_id": e["transaction_id"],
            "sender": e["sender"],
            "receiver": e["receiver"],
            "amount": e["amount"],
            "risk_level": e["dashboard_risk_level"],
            "original_gat_probability": e["gat_probability"],
            "human_decision": e.get("human_decision"),
            "status": e.get("investigation_status")
        })
    return {"nodes": nodes, "edges": edges}


@app.post("/api/investigations/{tx_id}/decision")
def record_investigation_decision(tx_id: str, payload: Dict[str, Any] = Body(default={} )):
    decision = str(payload.get("decision", "")).strip().upper()
    if decision in {"LEGITIMATE", "CLEAR", "APPROVE"}:
        return approve_investigation(tx_id, payload)
    if decision in {"CONFIRMED SUSPICIOUS", "CONFIRMED_SUSPICIOUS", "SUSPICIOUS", "CONFIRM"}:
        return confirm_investigation(tx_id, payload)
    raise HTTPException(status_code=400, detail="decision must be LEGITIMATE/CLEAR or CONFIRMED_SUSPICIOUS")

@app.post("/api/investigations/{tx_id}/approve")
def approve_investigation(tx_id: str, payload: Dict[str, Any] = Body(default={})):
    if not engine:
        raise HTTPException(status_code=500, detail="GAT Engine not initialized")
    
    found_tx = None
    for e in engine.edges:
        if e["transaction_id"] == tx_id:
            found_tx = e
            break
            
    if not found_tx:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")
    if not found_tx.get("fanout_alert"):
        raise HTTPException(status_code=409, detail="Human review is only available for fan-out alerts")
        
    found_tx["investigation_status"] = "RESOLVED"
    found_tx["human_decision"] = "LEGITIMATE"
    
    history_entry = {
        "transaction_id": tx_id,
        "timestamp": found_tx["timestamp"],
        "sender": found_tx["sender"],
        "receiver": found_tx["receiver"],
        "amount": found_tx["amount"],
        "original_gat_risk_level": found_tx["risk_level"],
        "original_gat_probability": found_tx["gat_probability"],
        "human_decision": "LEGITIMATE",
        "investigator": payload.get("investigator", "Human Auditor"),
        "decision_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "final_status": "RESOLVED / LEGITIMATE",
        "notes": payload.get("notes", "Marked as legitimate after review")
    }
    investigation_history_log.insert(0, history_entry)
    logger.info(f"Investigation Approved: {tx_id} marked LEGITIMATE by {history_entry['investigator']}")
    return {"status": "success", "transaction": found_tx, "history_entry": history_entry}

@app.post("/api/investigations/{tx_id}/confirm")
def confirm_investigation(tx_id: str, payload: Dict[str, Any] = Body(default={})):
    if not engine:
        raise HTTPException(status_code=500, detail="GAT Engine not initialized")
        
    found_tx = None
    for e in engine.edges:
        if e["transaction_id"] == tx_id:
            found_tx = e
            break
            
    if not found_tx:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")
    if not found_tx.get("fanout_alert"):
        raise HTTPException(status_code=409, detail="Human review is only available for fan-out alerts")
        
    found_tx["investigation_status"] = "RESOLVED"
    found_tx["human_decision"] = "CONFIRMED_SUSPICIOUS"
    
    history_entry = {
        "transaction_id": tx_id,
        "timestamp": found_tx["timestamp"],
        "sender": found_tx["sender"],
        "receiver": found_tx["receiver"],
        "amount": found_tx["amount"],
        "original_gat_risk_level": found_tx["risk_level"],
        "original_gat_probability": found_tx["gat_probability"],
        "human_decision": "CONFIRMED_SUSPICIOUS",
        "investigator": payload.get("investigator", "Human Auditor"),
        "decision_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "final_status": "CONFIRMED SUSPICIOUS",
        "notes": payload.get("notes", "Confirmed suspicious activity")
    }
    investigation_history_log.insert(0, history_entry)
    logger.info(f"Investigation Confirmed: {tx_id} confirmed SUSPICIOUS by {history_entry['investigator']}")
    return {"status": "success", "transaction": found_tx, "history_entry": history_entry}

@app.get("/api/investigations/history")
def get_investigation_history():
    return investigation_history_log

@app.get("/api/dashboard/summary")
def get_dashboard_summary():
    if not engine:
        return {
            "processed": 0, "gat_high": 0, "gat_medium": 0, "gat_low": 0,
            "fanout_candidates": 0, "fanout_alerts": 0, "high_fanout": 0,
            "medium_fanout": 0, "resolved": 0, "active_alerts_count": 0,
            "recent_transactions": [],
        }

    total = len(engine.edges)
    high = sum(1 for e in engine.edges if e["dashboard_risk_level"] == "HIGH")
    med = sum(1 for e in engine.edges if e["dashboard_risk_level"] == "MEDIUM")
    low = sum(1 for e in engine.edges if e["dashboard_risk_level"] == "LOW")
    resolved = len(investigation_history_log)
    gat_high = sum(1 for e in engine.edges if e.get("risk_level") == "HIGH")
    gat_medium = sum(1 for e in engine.edges if e.get("risk_level") == "MEDIUM")
    gat_low = sum(1 for e in engine.edges if e.get("risk_level") == "LOW")
    fanout_candidates = sum(1 for e in engine.edges if e.get("fanout_candidate"))
    fanout_alerts = sum(1 for e in engine.edges if e.get("fanout_alert"))
    high_fanout = sum(1 for e in engine.edges if e.get("alert_level") == "HIGH_FANOUT")
    medium_fanout = sum(1 for e in engine.edges if e.get("alert_level") == "MEDIUM_FANOUT")
    active_alerts = fanout_alerts
    recent = engine.edges[-30:][::-1]

    return {
        "processed": total,
        "high_risk": high,
        "medium_risk": med,
        "low_risk": low,
        "resolved": resolved,
        "active_alerts_count": active_alerts,
        "gat_high": gat_high,
        "gat_medium": gat_medium,
        "gat_low": gat_low,
        "fanout_candidates": fanout_candidates,
        "fanout_alerts": fanout_alerts,
        "high_fanout": high_fanout,
        "medium_fanout": medium_fanout,
        "recent_transactions": recent
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000, reload=False)
