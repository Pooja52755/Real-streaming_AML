-- ==============================================================================
-- AML FRAUD DETECTION SYSTEM: SUPABASE DATABASE INITIALIZATION SCRIPT
-- Run this in your Supabase SQL Editor (Dashboard -> SQL Editor -> New Query)
-- ==============================================================================

-- 1. Human Authorizer Decisions & Retraining Feedback Ledger
CREATE TABLE IF NOT EXISTS public.auditor_decisions (
    target_id TEXT PRIMARY KEY,
    transaction_id TEXT,
    gat_prob DOUBLE PRECISION DEFAULT 0.046942,
    risk_level TEXT DEFAULT 'High',
    human_decision TEXT DEFAULT 'Approve',
    decision TEXT DEFAULT 'Approve',
    training_label INTEGER DEFAULT 0,
    training_label_desc TEXT,
    remarks TEXT,
    notes TEXT,
    revision_remark TEXT,
    account TEXT,
    to_account TEXT,
    amount TEXT,
    timestamp TEXT,
    is_revised BOOLEAN DEFAULT false,
    revision_history JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_auditor_decisions_account ON public.auditor_decisions(account);
CREATE INDEX IF NOT EXISTS idx_auditor_decisions_decision ON public.auditor_decisions(decision);
CREATE INDEX IF NOT EXISTS idx_auditor_decisions_label ON public.auditor_decisions(training_label);

-- 2. Streaming & Testing Transactions Table (Source of Truth)
CREATE TABLE IF NOT EXISTS public.transactions (
    id BIGSERIAL PRIMARY KEY,
    tx_id TEXT UNIQUE NOT NULL,
    timestamp TEXT,
    from_account TEXT NOT NULL,
    to_account TEXT NOT NULL,
    amount DOUBLE PRECISION DEFAULT 0.0,
    currency TEXT DEFAULT 'US Dollar',
    payment_format TEXT DEFAULT 'Wire',
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transactions_from_acc ON public.transactions(from_account);
CREATE INDEX IF NOT EXISTS idx_transactions_to_acc ON public.transactions(to_account);
CREATE INDEX IF NOT EXISTS idx_transactions_ts ON public.transactions(timestamp);

-- 3. GAT Model Predictions & Risk Log Table
CREATE TABLE IF NOT EXISTS public.predictions (
    id BIGSERIAL PRIMARY KEY,
    tx_id TEXT NOT NULL,
    from_account TEXT NOT NULL,
    to_account TEXT NOT NULL,
    gat_prob DOUBLE PRECISION,
    risk_score INTEGER,
    risk_tier TEXT,
    is_fanout BOOLEAN DEFAULT false,
    timestamp TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_tx_id ON public.predictions(tx_id);

-- 4. Row-Level Security (RLS) Configuration
-- Enable public read/write via Anon / Service Role for Streamlit Cloud
ALTER TABLE public.auditor_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow anon read/write auditor_decisions" ON public.auditor_decisions;
CREATE POLICY "Allow anon read/write auditor_decisions" 
ON public.auditor_decisions FOR ALL 
TO anon, authenticated, service_role 
USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Allow anon read/write transactions" ON public.transactions;
CREATE POLICY "Allow anon read/write transactions" 
ON public.transactions FOR ALL 
TO anon, authenticated, service_role 
USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Allow anon read/write predictions" ON public.predictions;
CREATE POLICY "Allow anon read/write predictions" 
ON public.predictions FOR ALL 
TO anon, authenticated, service_role 
USING (true) WITH CHECK (true);
