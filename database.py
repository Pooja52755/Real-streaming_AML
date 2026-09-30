"""
Database Access Layer for AML Fraud Detection System
=====================================================
Clean database interface separating persistent data operations (Supabase PostgreSQL /
local fallback mirror) from PyG graph construction, GAT model inference, and Streamlit UI.

Provides:
- get_transactions(limit=1000)
- get_transaction(tx_id)
- get_historical_transactions(accounts, before_timestamp, window_hours)
- insert_transaction(tx_data)
- save_prediction(pred_data)
- save_auditor_decision(decision_data)
- get_auditor_decisions()
- get_auditor_decision(target_id)
- is_connected()
- get_status()
- migrate_seed_data_to_supabase()
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
from supabase_client import supabase_mgr, SupabaseManager, get_supabase_credentials


def get_transactions(limit: int = 1000) -> pd.DataFrame:
    """
    Retrieves all available transactions from Supabase 'transactions' table,
    falling back to local seed/demo CSV if Supabase is unconfigured or offline.
    """
    return supabase_mgr.get_transactions(limit=limit)


def get_transaction(tx_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single transaction by transaction ID.
    """
    return supabase_mgr.get_transaction(tx_id=tx_id)


def get_historical_transactions(
    accounts: Union[str, List[str]],
    before_timestamp: Optional[str] = None,
    window_hours: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves historical transactions involving the specified accounts.
    Used for dynamic PyG graph construction and 1-hop ego-network assembly.
    """
    return supabase_mgr.get_historical_transactions(
        accounts=accounts,
        before_timestamp=before_timestamp,
        window_hours=window_hours
    )


def insert_transaction(tx_data: Dict[str, Any]) -> bool:
    """
    Persists a new transaction into Supabase PostgreSQL.
    """
    return supabase_mgr.insert_transaction(tx_data=tx_data)


def save_prediction(pred_data: Dict[str, Any]) -> bool:
    """
    Logs GAT model inference output and risk score into Supabase.
    """
    return supabase_mgr.save_prediction(pred_data=pred_data)


def save_auditor_decision(record: Dict[str, Any]) -> bool:
    """
    Persists human authorizer compliance decisions (Approve, Reject, Escalate),
    audit notes, and timestamps to Supabase 'auditor_decisions' table.
    """
    return supabase_mgr.save_decision(record=record)


def get_auditor_decisions() -> Dict[str, Dict[str, Any]]:
    """
    Retrieves all auditor decisions from Supabase PostgreSQL.
    """
    return supabase_mgr.load_decisions()


def get_auditor_decision(target_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific authorizer decision by target_id.
    """
    return supabase_mgr.get_auditor_decision(target_id=target_id)


def is_connected() -> bool:
    """
    Returns True if Supabase PostgreSQL database is connected and active.
    """
    return supabase_mgr.is_connected()


def get_status() -> Dict[str, Any]:
    """
    Returns current database connectivity and source status.
    """
    return supabase_mgr.get_status()


def migrate_seed_data_to_supabase() -> int:
    """
    Seeds/migrates the 100 historical transactions into Supabase table.
    """
    return supabase_mgr.sync_csv_to_supabase()
