"""
Supabase Cloud Data & Storage Integration Layer for AML Fraud Detection
========================================================================
Supports:
1. Cloud persistence of human authorizer decisions in Supabase ('auditor_decisions' table).
2. Cloud persistence and retrieval of historical & streaming transactions ('transactions' table).
3. Cloud persistence of GAT model inference logs ('predictions' table).
4. Dynamic historical graph retrieval for ego-network reconstruction.
5. Seamless integration with Streamlit Cloud via st.secrets["SUPABASE_URL"] and st.secrets["SUPABASE_KEY"].
6. Resilient local fallback: if Supabase credentials are not configured or network is offline,
   transparently falls back to local storage (Data/auditor_decisions.json & Data/testing_trans.csv).
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import pandas as pd

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger("supabase_client")

# Local Fallback Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DECISIONS_PATH = os.path.join(BASE_DIR, "Data", "auditor_decisions.json")
LOCAL_TRANS_PATH = os.path.join(BASE_DIR, "Data", "testing_trans.csv")


def get_supabase_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Retrieves Supabase URL and API Key with priority:
    1. Streamlit Cloud Secrets (st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    2. OS Environment Variables (SUPABASE_URL, SUPABASE_KEY / SUPABASE_SERVICE_ROLE_KEY)
    """
    url = None
    key = None

    if STREAMLIT_AVAILABLE:
        try:
            if hasattr(st, "secrets"):
                url = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase_url")
                key = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase_key") or st.secrets.get("SUPABASE_ANON_KEY")
        except Exception:
            pass

    if not url:
        url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
    if not key:
        key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")

    if url:
        url = str(url).strip().rstrip("/")
    if key:
        key = str(key).strip()

    return url, key


class SupabaseManager:
    """Manages cloud sync for AML decisions, transactions, and historical graph context with Supabase."""
    
    _instance = None

    def __init__(self):
        self.url, self.key = get_supabase_credentials()
        self._connected = False
        self._cached_tx_df: Optional[pd.DataFrame] = None
        if self.url and self.key:
            self._check_connection()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_headers(self) -> Dict[str, str]:
        return {
            "apikey": self.key or "",
            "Authorization": f"Bearer {self.key or ''}",
            "Content-Type": "application/json",
            "Prefer": "return=representation,resolution=merge-duplicates"
        }

    def _check_connection(self) -> bool:
        if not self.url or not self.key or not REQUESTS_AVAILABLE:
            self._connected = False
            return False
        try:
            test_endpoint = f"{self.url}/rest/v1/auditor_decisions?select=target_id&limit=1"
            resp = requests.get(test_endpoint, headers=self._get_headers(), timeout=4)
            # 200/206 means table exists, 404 means table not created yet but endpoint alive
            if resp.status_code in [200, 206, 404]:
                self._connected = True
                return True
            else:
                self._connected = False
                return False
        except Exception as e:
            logger.warning(f"Supabase connection check failed: {e}")
            self._connected = False
            return False

    def is_connected(self) -> bool:
        """Returns True if valid Supabase credentials are provided and connected."""
        if not self.url or not self.key:
            self.url, self.key = get_supabase_credentials()
            if self.url and self.key:
                self._check_connection()
        return self._connected

    def get_status(self) -> Dict[str, Any]:
        """Provides status report for the UI and diagnostics."""
        connected = self.is_connected()
        masked_key = f"{self.key[:6]}...{self.key[-4:]}" if self.key and len(self.key) > 10 else "None"
        return {
            "connected": connected,
            "provider": "Supabase PostgreSQL Database" if connected else "Local Filesystem Fallback (Ready for Supabase)",
            "supabase_url": self.url or "Not Configured (st.secrets['SUPABASE_URL'])",
            "supabase_key": masked_key,
            "decisions_source": "Supabase 'auditor_decisions' table" if connected else "Data/auditor_decisions.json",
            "transactions_source": "Supabase 'transactions' table" if connected else "Data/testing_trans.csv"
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  TRANSACTIONS & HISTORICAL GRAPH QUERIES
    # ══════════════════════════════════════════════════════════════════════════

    def get_transactions(self, limit: int = 1000) -> pd.DataFrame:
        """
        Loads transactions from Supabase 'transactions' table.
        Falls back to local testing_trans.csv if Supabase is offline or empty.
        """
        if self.is_connected() and REQUESTS_AVAILABLE:
            try:
                endpoint = f"{self.url}/rest/v1/transactions?select=*&order=id.asc&limit={limit}"
                resp = requests.get(endpoint, headers=self._get_headers(), timeout=8)
                if resp.status_code == 200:
                    rows = resp.json()
                    if rows and len(rows) > 0:
                        df = pd.DataFrame(rows)
                        # Clean column mapping to match expected dataframe schema
                        col_rename = {}
                        if "from_account" in df.columns:
                            col_rename["from_account"] = "Account"
                        if "to_account" in df.columns:
                            col_rename["to_account"] = "Account.1"
                        if "amount" in df.columns:
                            col_rename["amount"] = "Amount Paid"
                        if "currency" in df.columns:
                            col_rename["currency"] = "Payment Currency"
                        if "payment_format" in df.columns:
                            col_rename["payment_format"] = "Payment Format"
                        if "timestamp" in df.columns:
                            col_rename["timestamp"] = "Timestamp"
                        df = df.rename(columns=col_rename)
                        for l_col in ["Is Laundering", "is_laundering", "Laundering"]:
                            if l_col in df.columns:
                                df = df.drop(columns=[l_col])
                        self._cached_tx_df = df
                        return df
            except Exception as e:
                logger.warning(f"Could not load transactions from Supabase: {e}")

        # Local Fallback
        if self._cached_tx_df is not None and not self._cached_tx_df.empty:
            return self._cached_tx_df

        if os.path.exists(LOCAL_TRANS_PATH):
            df = pd.read_csv(LOCAL_TRANS_PATH)
            for l_col in ["Is Laundering", "is_laundering", "Laundering"]:
                if l_col in df.columns:
                    df = df.drop(columns=[l_col])
            self._cached_tx_df = df
            return df

        return pd.DataFrame()

    def get_transaction(self, tx_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single transaction by its tx_id from Supabase or local store."""
        if not tx_id:
            return None
        tx_id_str = str(tx_id).strip()

        # 1. Try Supabase
        if self.is_connected() and REQUESTS_AVAILABLE:
            try:
                endpoint = f"{self.url}/rest/v1/transactions?tx_id=eq.{tx_id_str}&limit=1"
                resp = requests.get(endpoint, headers=self._get_headers(), timeout=5)
                if resp.status_code == 200:
                    rows = resp.json()
                    if rows:
                        return rows[0]
            except Exception as e:
                logger.warning(f"Supabase get_transaction failed: {e}")

        # 2. Local Fallback
        df = self.get_transactions()
        if not df.empty:
            # check by index or string representation
            if "tx_id" in df.columns:
                m = df[df["tx_id"].astype(str) == tx_id_str]
                if not m.empty:
                    return m.iloc[0].to_dict()
            # or by synthetic key TX-SIM-00001
            if tx_id_str.startswith("TX-SIM-"):
                try:
                    idx = int(tx_id_str.replace("TX-SIM-", "")) - 1
                    if 0 <= idx < len(df):
                        r = df.iloc[idx].to_dict()
                        r["tx_id"] = tx_id_str
                        return r
                except Exception:
                    pass

        return None

    def get_historical_transactions(
        self,
        accounts: Union[str, List[str]],
        before_timestamp: Optional[str] = None,
        window_hours: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant historical transactions involving the specified accounts.
        Used to dynamically reconstruct ego-network subgraphs from Supabase for PyG model inference.
        """
        if isinstance(accounts, str):
            acc_list = [accounts.strip()]
        else:
            acc_list = [str(a).strip() for a in accounts if a]

        if not acc_list:
            return []

        results: List[Dict[str, Any]] = []

        # 1. Supabase Query
        if self.is_connected() and REQUESTS_AVAILABLE:
            try:
                acc_filter_str = ",".join(acc_list)
                endpoint = f"{self.url}/rest/v1/transactions?or=(from_account.in.({acc_filter_str}),to_account.in.({acc_filter_str}))&order=timestamp.asc&limit=500"
                resp = requests.get(endpoint, headers=self._get_headers(), timeout=6)
                if resp.status_code == 200:
                    raw_rows = resp.json()
                    for r in raw_rows:
                        # Filter by timestamp if provided
                        r_ts = str(r.get("timestamp", ""))
                        if before_timestamp and r_ts > str(before_timestamp):
                            continue
                        results.append({
                            "tx_id": r.get("tx_id", f"TX-SUPA-{r.get('id', 0)}"),
                            "from_account": r.get("from_account"),
                            "to_account": r.get("to_account"),
                            "amount": float(r.get("amount", 0.0)),
                            "currency": r.get("currency", "US Dollar"),
                            "payment_format": r.get("payment_format", "Wire"),
                            "timestamp": r_ts
                        })
                    if results:
                        return results
            except Exception as e:
                logger.warning(f"Error querying historical transactions from Supabase: {e}")

        # 2. Local Fallback
        df = self.get_transactions()
        if not df.empty:
            acc_col = "Account" if "Account" in df.columns else "From Account"
            to_col = "Account.1" if "Account.1" in df.columns else "To Account"
            amt_col = "Amount Paid" if "Amount Paid" in df.columns else "Amount"
            curr_col = "Payment Currency" if "Payment Currency" in df.columns else "Currency"
            fmt_col = "Payment Format" if "Payment Format" in df.columns else "Format"
            ts_col = "Timestamp" if "Timestamp" in df.columns else "timestamp"

            matches = df[(df[acc_col].astype(str).isin(acc_list)) | (df[to_col].astype(str).isin(acc_list))]
            for idx, r in matches.iterrows():
                r_ts = str(r.get(ts_col, ""))
                if before_timestamp and r_ts > str(before_timestamp):
                    continue
                results.append({
                    "tx_id": f"TX-SIM-{idx+1:05d}",
                    "from_account": str(r.get(acc_col, "")),
                    "to_account": str(r.get(to_col, "")),
                    "amount": float(r.get(amt_col, 0.0)),
                    "currency": str(r.get(curr_col, "US Dollar")),
                    "payment_format": str(r.get(fmt_col, "Wire")),
                    "timestamp": r_ts
                })

        return results

    def insert_transaction(self, tx_data: Dict[str, Any]) -> bool:
        """Inserts a new transaction into the Supabase 'transactions' table."""
        if self.is_connected() and REQUESTS_AVAILABLE:
            try:
                endpoint = f"{self.url}/rest/v1/transactions"
                payload = {
                    "tx_id": str(tx_data.get("tx_id", f"TX-MANUAL-{int(datetime.now().timestamp())}")),
                    "timestamp": str(tx_data.get("timestamp", datetime.now().isoformat())),
                    "from_account": str(tx_data.get("from_account", tx_data.get("Account", ""))),
                    "to_account": str(tx_data.get("to_account", tx_data.get("Account.1", ""))),
                    "amount": float(tx_data.get("amount", tx_data.get("Amount Paid", 0.0))),
                    "currency": str(tx_data.get("currency", tx_data.get("Payment Currency", "US Dollar"))),
                    "payment_format": str(tx_data.get("payment_format", tx_data.get("Payment Format", "Wire")))
                }
                resp = requests.post(endpoint, json=payload, headers=self._get_headers(), timeout=5)
                return resp.status_code in [200, 201, 204]
            except Exception as e:
                logger.warning(f"Error inserting transaction to Supabase: {e}")
                return False
        return False

    def sync_csv_to_supabase(self, csv_path: str = LOCAL_TRANS_PATH) -> int:
        """Uploads and seeds the 100 testing CSV rows into Supabase 'transactions' table."""
        if not self.is_connected() or not os.path.exists(csv_path) or not REQUESTS_AVAILABLE:
            return 0
        try:
            df = pd.read_csv(csv_path)
            rows_to_insert = []
            for idx, r in df.iterrows():
                rows_to_insert.append({
                    "tx_id": f"TX-SIM-{idx+1:05d}",
                    "timestamp": str(r.get("Timestamp", "")),
                    "from_account": str(r.get("Account", r.get("From Account", ""))),
                    "to_account": str(r.get("Account.1", r.get("To Account", ""))),
                    "amount": float(r.get("Amount Paid", r.get("Amount", 0.0))),
                    "currency": str(r.get("Payment Currency", r.get("Currency", "US Dollar"))),
                    "payment_format": str(r.get("Payment Format", "Wire")),
                })

            endpoint = f"{self.url}/rest/v1/transactions"
            resp = requests.post(endpoint, json=rows_to_insert, headers=self._get_headers(), timeout=12)
            if resp.status_code in [200, 201, 204]:
                logger.info(f"Successfully migrated {len(rows_to_insert)} transactions to Supabase!")
                return len(rows_to_insert)
            else:
                logger.warning(f"Supabase transactions upload returned {resp.status_code}: {resp.text}")
                return 0
        except Exception as e:
            logger.warning(f"Error syncing CSV to Supabase: {e}")
            return 0

    # ══════════════════════════════════════════════════════════════════════════
    #  PREDICTIONS PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════

    def save_prediction(self, pred_data: Dict[str, Any]) -> bool:
        """Logs GAT inference prediction to Supabase 'predictions' table."""
        if not self.is_connected() or not REQUESTS_AVAILABLE:
            return False
        try:
            endpoint = f"{self.url}/rest/v1/predictions"
            payload = {
                "tx_id": str(pred_data.get("tx_id", "")),
                "from_account": str(pred_data.get("from_account", "")),
                "to_account": str(pred_data.get("to_account", "")),
                "gat_prob": float(pred_data.get("gat_prob", 0.0)),
                "risk_score": int(pred_data.get("risk_score", 0)),
                "risk_tier": str(pred_data.get("risk_tier", "Low")),
                "is_fanout": bool(pred_data.get("is_fanout", False))
            }
            resp = requests.post(endpoint, json=payload, headers=self._get_headers(), timeout=4)
            return resp.status_code in [200, 201, 204]
        except Exception:
            return False

    # ══════════════════════════════════════════════════════════════════════════
    #  HUMAN AUTHORIZER DECISIONS (Supabase Persistent Truth + Local Mirror)
    # ══════════════════════════════════════════════════════════════════════════

    def load_decisions(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads all auditor decisions from Supabase.
        Falls back to Data/auditor_decisions.json if Supabase is offline or unconfigured.
        """
        decisions: Dict[str, Dict[str, Any]] = {}

        # 1. Try Supabase first if connected
        if self.is_connected() and REQUESTS_AVAILABLE:
            try:
                endpoint = f"{self.url}/rest/v1/auditor_decisions?select=*"
                resp = requests.get(endpoint, headers=self._get_headers(), timeout=6)
                if resp.status_code == 200:
                    rows = resp.json()
                    for r in rows:
                        tid = r.get("target_id") or r.get("transaction_id")
                        if tid:
                            decisions[tid] = r
                    # Sync local file with cloud state
                    if decisions:
                        try:
                            with open(LOCAL_DECISIONS_PATH, "w", encoding="utf-8") as f:
                                json.dump(decisions, f, indent=2)
                        except Exception:
                            pass
                    return decisions
            except Exception as e:
                logger.warning(f"Error loading decisions from Supabase: {e}")

        # 2. Local Fallback
        if os.path.exists(LOCAL_DECISIONS_PATH):
            try:
                with open(LOCAL_DECISIONS_PATH, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        decisions = json.loads(content)
            except Exception as e:
                logger.warning(f"Error loading local decisions file: {e}")

        return decisions

    def get_auditor_decision(self, target_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single decision for target_id from Supabase or local cache."""
        decs = self.load_decisions()
        return decs.get(target_id)

    def save_decision(self, record: Dict[str, Any]) -> bool:
        """
        Saves or updates a human decision in Supabase (upsert) and mirrors to local JSON.
        """
        target_id = record.get("target_id") or record.get("transaction_id")
        if not target_id:
            return False

        # 1. Mirror locally first
        local_saved = False
        try:
            current_decisions = {}
            if os.path.exists(LOCAL_DECISIONS_PATH):
                with open(LOCAL_DECISIONS_PATH, "r", encoding="utf-8") as f:
                    c = f.read().strip()
                    if c:
                        current_decisions = json.loads(c)
            current_decisions[target_id] = record
            with open(LOCAL_DECISIONS_PATH, "w", encoding="utf-8") as f:
                json.dump(current_decisions, f, indent=2)
            local_saved = True
        except Exception as e:
            logger.warning(f"Error saving decision locally: {e}")

        # 2. Upsert to Supabase
        if self.is_connected() and REQUESTS_AVAILABLE:
            try:
                endpoint = f"{self.url}/rest/v1/auditor_decisions"
                payload = {
                    "target_id": str(target_id),
                    "transaction_id": str(record.get("transaction_id", target_id)),
                    "gat_prob": float(record.get("gat_prob", 0.046942)),
                    "risk_level": str(record.get("risk_level", "High")),
                    "human_decision": str(record.get("human_decision", record.get("decision", "Approve"))),
                    "decision": str(record.get("decision", "Approve")),
                    "training_label": int(record.get("training_label", 0)),
                    "training_label_desc": str(record.get("training_label_desc", "")),
                    "remarks": str(record.get("remarks", record.get("notes", ""))),
                    "notes": str(record.get("notes", "")),
                    "revision_remark": str(record.get("revision_remark", "")),
                    "account": str(record.get("account", "")),
                    "to_account": str(record.get("to_account", "")),
                    "amount": str(record.get("amount", "")),
                    "timestamp": str(record.get("timestamp", "")),
                    "is_revised": bool(record.get("is_revised", False)),
                    "revision_history": record.get("revision_history", [])
                }
                headers = self._get_headers()
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=6)
                if resp.status_code in [200, 201, 204]:
                    return True
                else:
                    logger.warning(f"Supabase upsert status {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.warning(f"Error upserting decision to Supabase: {e}")

        return local_saved


# Singleton accessor
supabase_mgr = SupabaseManager.get_instance()
