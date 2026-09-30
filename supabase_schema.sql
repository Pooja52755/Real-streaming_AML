-- ==============================================================================
-- AML FRAUD DETECTION SYSTEM: FULL SUPABASE DATABASE SCHEMA & SEED DATA
-- Run this entire script in your Supabase SQL Editor (Dashboard -> SQL Editor -> New Query)
-- ==============================================================================

-- 1. Accounts Metadata Table
CREATE TABLE IF NOT EXISTS public.accounts (
    account_number TEXT PRIMARY KEY,
    bank_name TEXT DEFAULT 'Global Bank',
    bank_id TEXT DEFAULT 'BNK-001',
    entity_id TEXT,
    entity_name TEXT,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Streaming & Testing Transactions Table (Persistent Source of Truth)
CREATE TABLE IF NOT EXISTS public.transactions (
    id BIGSERIAL PRIMARY KEY,
    tx_id TEXT UNIQUE NOT NULL,
    timestamp TEXT,
    from_bank TEXT,
    from_account TEXT NOT NULL,
    to_bank TEXT,
    to_account TEXT NOT NULL,
    amount DOUBLE PRECISION DEFAULT 0.0,
    amount_received DOUBLE PRECISION DEFAULT 0.0,
    receiving_currency TEXT DEFAULT 'US Dollar',
    currency TEXT DEFAULT 'US Dollar',
    payment_format TEXT DEFAULT 'Wire',
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Dynamic Graph Topology & Historical Edges Table
CREATE TABLE IF NOT EXISTS public.graph_edges (
    id BIGSERIAL PRIMARY KEY,
    tx_id TEXT,
    from_account TEXT NOT NULL,
    to_account TEXT NOT NULL,
    amount DOUBLE PRECISION DEFAULT 0.0,
    currency TEXT DEFAULT 'US Dollar',
    payment_format TEXT DEFAULT 'Wire',
    timestamp TEXT,
    hop INTEGER DEFAULT 1,
    is_fanout BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 4. Human Authorizer Decisions & Retraining Feedback Ledger
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

-- 5. GAT Model Predictions Log Table
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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_from_acc ON public.transactions(from_account);
CREATE INDEX IF NOT EXISTS idx_transactions_to_acc ON public.transactions(to_account);
CREATE INDEX IF NOT EXISTS idx_transactions_ts ON public.transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_graph_edges_from ON public.graph_edges(from_account);
CREATE INDEX IF NOT EXISTS idx_graph_edges_to ON public.graph_edges(to_account);
CREATE INDEX IF NOT EXISTS idx_auditor_decisions_acc ON public.auditor_decisions(account);

-- Row-Level Security (RLS) Configuration
ALTER TABLE public.accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.graph_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.auditor_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow anon accounts" ON public.accounts;
CREATE POLICY "Allow anon accounts" ON public.accounts FOR ALL TO anon, authenticated, service_role USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Allow anon transactions" ON public.transactions;
CREATE POLICY "Allow anon transactions" ON public.transactions FOR ALL TO anon, authenticated, service_role USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Allow anon graph_edges" ON public.graph_edges;
CREATE POLICY "Allow anon graph_edges" ON public.graph_edges FOR ALL TO anon, authenticated, service_role USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Allow anon decisions" ON public.auditor_decisions;
CREATE POLICY "Allow anon decisions" ON public.auditor_decisions FOR ALL TO anon, authenticated, service_role USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Allow anon predictions" ON public.predictions;
CREATE POLICY "Allow anon predictions" ON public.predictions FOR ALL TO anon, authenticated, service_role USING (true) WITH CHECK (true);

-- ==============================================================================
-- SEED DATA: 134 ACCOUNT PROFILES (ON CONFLICT DO NOTHING)
-- ==============================================================================
INSERT INTO public.accounts (account_number, bank_name, bank_id, entity_id, entity_name) VALUES
('8088DA970', 'Spain Bank #305', '215820', '800B11AC0', 'Sole Proprietorship #31565'),
('80AD106C0', 'National Bank of Milford', '8814', '8005EC0E0', 'Partnership #53578'),
('8163227C0', 'Germany Bank #1213', '234539', '800C2E830', 'Corporation #57597'),
('800804470', 'National Bank of Seattle', '224', '80155E7F0', 'Corporation #1175'),
('80C58FC30', 'Austria Bank #101', '33067', '800C1A490', 'Corporation #34259'),
('801426610', 'Brook Savings Bank', '13354', '80026AB40', 'Sole Proprietorship #13148'),
('809713AD0', 'France Bank #43', '115332', '800B04FC0', 'Corporation #33089'),
('803328260', 'Spain Bank #72', '25634', '8009F9B80', 'Corporation #30742'),
('800822070', 'Oasis Credit Union', '1291', '801296080', 'Sole Proprietorship #1294'),
('8075299E0', 'Capital Trust Bank', '1110', '800115680', 'Sole Proprietorship #10852'),
('81950BA71', 'Switzerland Bank #16', '219', '8015B10D0', 'Partnership #953'),
('800843180', 'Germany Bank #15', '2514', '800860300', 'Corporation #29016'),
('800D10AF0', 'Arbor Community Bank', '21414', '80015CCC0', 'Corporation #2039'),
('80B6F24C0', 'Germany Bank #2100', '225265', '800BBAE30', 'Sole Proprietorship #32639'),
('81C0115A1', 'First Bank of Danbury', '20', '800773900', 'Partnership #50310'),
('8009B5440', 'National Bank of Philadelphia', '1231', '8000A44E0', 'Partnership #1567'),
('802DC7DC0', 'First Bank of Huron', '2692', '800233C40', 'Partnership #5300'),
('80E119340', 'National Bank of Pittsburgh', '16866', '801232B50', 'Sole Proprietorship #24669'),
('800404F90', 'Greece Bank #8', '600', '800827C70', 'Partnership #14778'),
('80F49DC80', 'Australia Bank #58', '41335', '8011C55F0', 'Partnership #61043'),
('8061DA640', 'National Bank of Boston', '215456', '8004A9D40', 'Corporation #9252'),
('800231EA0', 'National Bank of Philadelphia', '1231', '8000A1900', 'Sole Proprietorship #344'),
('800618C30', 'First Bank of Portland', '1', '8000649C0', 'Corporation #773'),
('8004943A0', 'Germany Bank #34', '423', '800BDCA70', 'Sole Proprietorship #7648'),
('8005150D0', 'Savings Bank of Detroit', '394', '8012DD690', 'Partnership #657'),
('805D120A0', 'Netherlands Bank #448', '114948', '800AF4300', 'Corporation #31482'),
('80009B8E0', 'Canada Bank #0', '8', '80104B200', 'Sole Proprietorship #65855'),
('800F36730', 'First Bank of Danbury', '21939', '8012751A0', 'Partnership #2511'),
('801C37770', 'Spain Bank #12', '21876', '8008D86C0', 'Corporation #31734'),
('800297990', 'The Pine Thrift', '349', '8012C26D0', 'Corporation #287'),
('80011D080', 'First Bank of Danbury', '20', '800055940', 'Sole Proprietorship #89'),
('800AF0150', 'Savings Bank of Miami', '11701', '80017B220', 'Partnership #1804'),
('80011E2A0', 'First Bank of Danbury', '20', '8012985E0', 'Partnership #94'),
('8008FF250', 'Flagstone Thrift', '11265', '8001321A0', 'Sole Proprietorship #1472'),
('80009FC90', 'Canada Bank #0', '8', '801093680', 'Partnership #44630'),
('80BA9F560', 'Japan Bank #42', '18', '80101CA80', 'Sole Proprietorship #44032'),
('8050C86F0', 'Bank of Detroit', '15640', '800326520', 'Corporation #7682'),
('805756EB0', 'Sea Savings Bank', '11968', '8001ABEE0', 'Sole Proprietorship #8030'),
('800AA1DA0', 'National Bank of Columbus', '12', '80007FB50', 'Partnership #53331'),
('81A217C11', 'Crytpo Bank #50', '169671', '8004515A0', 'Sole Proprietorship #52590'),
('800AAE3B0', 'Germany Bank #34', '423', '80083FC60', 'Partnership #53993'),
('800323610', 'Italy Bank #18', '0', '800AAB280', 'Partnership #25326'),
('800DAA850', 'Netherlands Bank #27', '498', '800856320', 'Partnership #31380'),
('8057B5070', 'National Bank of Billings', '531', '800EBC480', 'Sole Proprietorship #8191'),
('8000A5A20', 'Savings Bank of Madison', '11', '8010992F0', 'Partnership #66806'),
('80F98ECB0', 'First Bank of Butte', '5394', '8003188A0', 'Partnership #74617'),
('800DF9310', 'Savings Bank of the North', '11255', '80012B100', 'Partnership #2308'),
('8000A8A80', 'Mexico Bank #4', '19', '80111ED00', 'Corporation #47126'),
('80B4A8A90', 'Japan Bank #42', '18', '80101BAC0', 'Sole Proprietorship #57658'),
('81A41D1E1', 'Crytpo Bank #50', '169671', '800C7C7D0', 'Partnership #50308'),
('8061FBE10', 'National Bank of Watertown', '23041', '800259DA0', 'Partnership #9058'),
('801B16880', 'Italy Bank #113', '14195', '800951F40', 'Partnership #66572'),
('80060B3A0', 'Bank of Philadelphia', '1217', '8014834B0', 'Sole Proprietorship #882'),
('801618FE0', 'Portugal Bank #122', '22264', '800C85270', 'Partnership #26651'),
('805FEC910', 'National Bank of Albany', '2175', '8001D5100', 'Sole Proprietorship #12122'),
('8033AFA50', 'China Bank #49', '27', '800CE4040', 'Corporation #39599'),
('802ED3AC0', 'Germany Bank #10', '15461', '8009CBD60', 'Corporation #51899'),
('8015879D0', 'Italy Bank #127', '1720', '800CA88B0', 'Sole Proprietorship #36113'),
('800E33990', 'First Bank of Danbury', '20', '80004ED10', 'Sole Proprietorship #2662'),
('80061FEF0', 'First Bank of Hartford', '1422', '8000D73A0', 'Corporation #56963'),
('800234E30', 'First Bank of Danbury', '20', '800949620', 'Sole Proprietorship #25229'),
('8095E9C40', 'Slovenia Bank #1285', '19419', '800AB6740', 'Sole Proprietorship #31852'),
('80B41F760', 'Bank of Lincoln', '222433', '801251F30', 'Partnership #3520'),
('8002BA9A0', 'First Bank of Danbury', '20', '800EEC550', 'Partnership #277'),
('81495FDF0', 'Savings Bank of the East', '40104', '80072A000', 'Partnership #21433'),
('817F979B0', 'Saudi Arabia Bank #37', '263574', '8013AEEE0', 'Sole Proprietorship #49939'),
('819B844C0', 'Bank of Seattle', '67359', '8007F2FF0', 'Partnership #66561'),
('801717B00', 'Germany Bank #121', '1390', '8008BCDC0', 'Corporation #29005'),
('800A53100', 'Savings Bank of Providence', '11770', '800182800', 'Sole Proprietorship #61132'),
('80081A660', 'First Bank of Huron', '1277', '800E70C80', 'Corporation #1166'),
('80B84CB60', 'Spruce Bank', '118314', '8012051E0', 'Partnership #11307'),
('80048A6E0', 'Slovakia Bank #45', '1776', '800B65920', 'Sole Proprietorship #23796'),
('800DE3000', 'First Bank of the South', '1120', '80011A320', 'Sole Proprietorship #2409'),
('80EDA5E40', 'Germany Bank #3', '23726', '80093C920', 'Partnership #72518'),
('819F8F421', 'Crytpo Bank #22', '170412', '8015B10D0', 'Partnership #953'),
('80D3D90C0', 'Canada Bank #43', '136009', '8010C6560', 'Partnership #71658'),
('80B56A940', 'National Bank of Denver', '111913', '800425040', 'Corporation #15232'),
('800774890', 'France Bank #7', '1595', '800B802E0', 'Partnership #31652'),
('80B9E6910', 'Netherlands Bank #226', '218587', '800B4F8C0', 'Partnership #27985'),
('805F5B360', 'France Bank #53', '16934', '800A77000', 'Corporation #29737'),
('8084D0590', 'Germany Bank #418', '21316', '800B8B540', 'Sole Proprietorship #31453'),
('804C8AB80', 'China Bank #44', '112646', '800DFA2B0', 'Partnership #38442'),
('80DCF61B0', 'National Bank of Albany', '18326', '8005255E0', 'Partnership #17016'),
('80ABCE300', 'India Bank #124', '127152', '800FD9530', 'Partnership #42993'),
('80B880140', 'Bank of Denver', '27009', '800F29190', 'Sole Proprietorship #4657'),
('8010542C0', 'Plateau Savings Bank', '21260', '800274940', 'Partnership #12618'),
('800DFB910', 'Germany Bank #19', '1768', '80088B280', 'Sole Proprietorship #59483'),
('80B42AB80', 'Sea Bank', '11495', '8009219E0', 'Partnership #7358'),
('8001505C0', 'First Bank of Danbury', '20', '800055E80', 'Corporation #28141'),
('81897DD10', 'Saudi Arabia Bank #187', '165362', '801436E20', 'Corporation #8623'),
('800A1DAE0', 'Germany Bank #15', '2514', '800860780', 'Corporation #28505'),
('8128067C0', 'Brazil Bank #20', '248739', '801093480', 'Sole Proprietorship #48001'),
('8116929A0', 'France Bank #105', '14652', '800A674A0', 'Partnership #35317'),
('8173C5700', 'Oasis Credit Union', '119593', '80054F7E0', 'Corporation #24835'),
('80005CA70', 'First Bank of Portland', '1', '800062380', 'Sole Proprietorship #55'),
('814214A40', 'Italy Bank #94', '35075', '800C31E30', 'Corporation #56639'),
('8054F7BB0', 'China Bank #213', '13577', '800CFBE40', 'Partnership #37107'),
('80BF623F0', 'China Bank #49', '27', '801012350', 'Sole Proprietorship #44294'),
('8003CF520', 'First Bank of Hartford', '1422', '801554850', 'Corporation #832'),
('8040E5E70', 'France Bank #52', '12503', '800914E60', 'Partnership #26775'),
('8132E5060', 'Savings Bank of Madison', '11', '80122AE90', 'Sole Proprietorship #4833'),
('803526830', 'Bank of Phoenix', '22547', '800225F80', 'Sole Proprietorship #6017'),
('800469B70', 'Netherlands Bank #27', '498', '800BF2170', 'Partnership #33241'),
('80224E670', 'Germany Bank #56', '5383', '800B1E1A0', 'Corporation #32612'),
('81A4AB801', 'Crytpo Bank #22', '170412', '8006D12E0', 'Partnership #50307'),
('80EA08F30', 'Savings Bank of New York', '24063', '8013FD020', 'Corporation #11677'),
('808A286D0', 'Italy Bank #47', '2557', '8008B08A0', 'Corporation #35661'),
('808B6C3B0', 'India Bank #49', '123260', '800F328F0', 'Corporation #43625'),
('811FC4DF0', 'National Bank of Lincoln', '1439', '801298760', 'Corporation #10619'),
('80FE82550', 'Germany Bank #285', '232814', '800B0DBA0', 'Partnership #55368'),
('800D5D510', 'Portugal Bank #122', '22264', '800C84E50', 'Partnership #54217'),
('818D46070', 'Saudi Arabia Bank #37', '263574', '80141E320', 'Sole Proprietorship #13267'),
('80C6EBF80', 'UK Bank #12', '29', '80105CF10', 'Corporation #10922'),
('800C6B2D0', 'First Bank of Danbury', '20', '80005B160', 'Corporation #1942'),
('8060D6D10', 'Croatia Bank #20', '5981', '800A04660', 'Partnership #33661'),
('800AECC10', 'Savings Bank of Madison', '11', '8000736B0', 'Sole Proprietorship #1852'),
('8064D3510', 'Golden Savings Bank', '116703', '8004D7120', 'Corporation #9580'),
('800245AD0', 'Oasis Credit Union', '1291', '8000ACC60', 'Partnership #315'),
('817F973B0', 'Saudi Arabia Bank #49', '65300', '8013C10C0', 'Corporation #49484'),
('809B05D90', 'India Bank #80', '24202', '800EF72A0', 'Corporation #43811'),
('80077B760', 'The Pine Thrift', '349', '8012F3270', 'Sole Proprietorship #5843'),
('800073410', 'China Bank #5', '15', '800819180', 'Sole Proprietorship #27215'),
('8004F4B00', 'First Bank of Danbury', '20', '8000582E0', 'Partnership #622'),
('8001744F0', 'National Bank of Columbus', '12', '80007C370', 'Sole Proprietorship #182'),
('802B12070', 'Italy Bank #529', '15516', '8009D6900', 'Corporation #30252'),
('802CCF530', 'China Bank #37', '5893', '800D1A160', 'Corporation #39635'),
('80F9F98C0', 'Australia Bank #171', '45104', '8011C4810', 'Sole Proprietorship #168'),
('8005F8370', 'Italy Bank #18', '0', '800988E60', 'Sole Proprietorship #25338'),
('800278A90', 'Slovakia Bank #4', '14', '80081A320', 'Sole Proprietorship #27161'),
('8070C98A0', 'Bank of Portland', '22129', '8001E3120', 'Corporation #72439'),
('8003EA620', 'National Bank of Billings', '531', '8000DFB80', 'Corporation #432'),
('8007224B0', 'National Bank of Philadelphia', '11047', '801266FC0', 'Sole Proprietorship #13526'),
('80227D060', 'Germany Bank #135', '4308', '800976D00', 'Sole Proprietorship #28686'),
('8014FCB10', 'First Bank of Danbury', '20', '800836700', 'Partnership #27350')
ON CONFLICT (account_number) DO NOTHING;

-- ==============================================================================
-- SEED DATA: 100 TRANSACTIONS (ON CONFLICT DO NOTHING)
-- ==============================================================================
INSERT INTO public.transactions (tx_id, timestamp, from_bank, from_account, to_bank, to_account, amount, amount_received, receiving_currency, currency, payment_format) VALUES
('TX-SIM-00001', '2022/09/01 00:03', '65300', '817F973B0', '263574', '817F979B0', 5684571.0, 5684571.0, 'Saudi Riyal', 'Saudi Riyal', 'Cheque'),
('TX-SIM-00002', '2022/09/01 09:12', '4308', '80227D060', '2557', '808A286D0', 72.63, 72.63, 'Euro', 'Euro', 'Cheque'),
('TX-SIM-00003', '2022/09/01 18:30', '14195', '801B16880', '234539', '8163227C0', 623.67, 623.67, 'Euro', 'Euro', 'Cash'),
('TX-SIM-00004', '2022/09/01 20:06', '1217', '80060B3A0', '22264', '800D5D510', 12196.07, 12196.07, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00005', '2022/09/01 20:28', '423', '8004943A0', '12', '800AA1DA0', 18534.45, 18534.45, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00006', '2022/09/01 20:55', '169671', '81A217C11', '170412', '81A4AB801', 0.276828, 0.276828, 'Bitcoin', 'Bitcoin', 'Bitcoin'),
('TX-SIM-00007', '2022/09/01 23:18', '215820', '8088DA970', '23726', '80EDA5E40', 16505.05, 16505.05, 'Euro', 'Euro', 'Credit Card'),
('TX-SIM-00008', '2022/09/02 01:54', '22547', '803526830', '40104', '81495FDF0', 3715.38, 3715.38, 'US Dollar', 'US Dollar', 'Credit Card'),
('TX-SIM-00009', '2022/09/02 06:37', '225265', '80B6F24C0', '35075', '814214A40', 885.46, 885.46, 'Euro', 'Euro', 'Cheque'),
('TX-SIM-00010', '2022/09/02 10:44', '11', '8000A5A20', '136009', '80D3D90C0', 29.34, 29.34, 'Canadian Dollar', 'Canadian Dollar', 'Credit Card'),
('TX-SIM-00011', '2022/09/02 11:06', '423', '8004943A0', '2514', '800843180', 1850.47, 1850.47, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00012', '2022/09/02 12:12', '423', '8004943A0', '349', '800297990', 16820.62, 16820.62, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00013', '2022/09/02 14:03', '2175', '805FEC910', '215456', '8061DA640', 52.17, 52.17, 'US Dollar', 'US Dollar', 'Credit Card'),
('TX-SIM-00014', '2022/09/02 16:29', '1277', '80081A660', '11265', '8008FF250', 93.86, 93.86, 'US Dollar', 'US Dollar', 'Cheque'),
('TX-SIM-00015', '2022/09/02 17:36', '1217', '80060B3A0', '224', '800804470', 5260.9, 5260.9, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00016', '2022/09/02 17:47', '423', '8004943A0', '20', '800C6B2D0', 19765.07, 19765.07, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00017', '2022/09/02 18:32', '170412', '819F8F421', '20', '81C0115A1', 0.964356, 0.964356, 'Bitcoin', 'Bitcoin', 'Bitcoin'),
('TX-SIM-00018', '2022/09/02 18:36', '25634', '803328260', '12503', '8040E5E70', 251.74, 251.74, 'Euro', 'Euro', 'Credit Card'),
('TX-SIM-00019', '2022/09/02 20:56', '1217', '80060B3A0', '1', '80005CA70', 5907.18, 5907.18, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00020', '2022/09/02 21:23', '123260', '808B6C3B0', '127152', '80ABCE300', 16640.39, 16640.39, 'Rupee', 'Rupee', 'Cheque'),
('TX-SIM-00021', '2022/09/03 07:38', '423', '8004943A0', '600', '800404F90', 15738.94, 15738.94, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00022', '2022/09/03 07:49', '1217', '80060B3A0', '20', '800234E30', 15249.24, 15249.24, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00023', '2022/09/03 08:21', '423', '8004943A0', '20', '800E33990', 10707.55, 10707.55, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00024', '2022/09/03 12:58', '1217', '80060B3A0', '20', '8004F4B00', 6904.98, 6904.98, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00025', '2022/09/03 14:50', '423', '8004943A0', '423', '800AAE3B0', 11331.44, 11331.44, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00026', '2022/09/03 16:48', '423', '8004943A0', '20', '80011D080', 12780.33, 12780.33, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00027', '2022/09/03 16:50', '1217', '80060B3A0', '11047', '8007224B0', 8628.31, 8628.31, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00028', '2022/09/03 17:36', '423', '8004943A0', '2514', '800A1DAE0', 2659.52, 2659.52, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00029', '2022/09/03 19:57', '423', '8004943A0', '1720', '8015879D0', 8172.36, 8172.36, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00030', '2022/09/03 21:00', '1217', '80060B3A0', '19', '8000A8A80', 12608.82, 12608.82, 'Australian Dollar', 'Australian Dollar', 'ACH'),
('TX-SIM-00031', '2022/09/03 23:20', '1217', '80060B3A0', '1', '800618C30', 14751.47, 14751.47, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00032', '2022/09/04 02:17', '423', '8004943A0', '11770', '800A53100', 13917.05, 13917.05, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00033', '2022/09/04 04:08', '219', '81950BA71', '169671', '81A41D1E1', 0.010015, 0.010015, 'Bitcoin', 'Bitcoin', 'Bitcoin'),
('TX-SIM-00034', '2022/09/04 05:30', '5981', '8060D6D10', '218587', '80B9E6910', 460.47, 460.47, 'Euro', 'Euro', 'Cash'),
('TX-SIM-00035', '2022/09/04 08:36', '423', '8004943A0', '1776', '80048A6E0', 4394.77, 4394.77, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00036', '2022/09/04 12:35', '1217', '80060B3A0', '11701', '800AF0150', 11470.47, 11470.47, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00037', '2022/09/04 13:33', '423', '8004943A0', '15', '800073410', 17846.21, 17846.21, 'Yuan', 'Yuan', 'ACH'),
('TX-SIM-00038', '2022/09/04 13:56', '1217', '80060B3A0', '20', '8001505C0', 2848.66, 2848.66, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00039', '2022/09/04 19:41', '11968', '805756EB0', '5394', '80F98ECB0', 7009.17, 7009.17, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00040', '2022/09/04 23:17', '222433', '80B41F760', '11495', '80B42AB80', 3901.34, 3901.34, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00041', '2022/09/05 07:58', '18', '80B4A8A90', '18', '80BA9F560', 20126.46, 20126.46, 'Ruble', 'Ruble', 'Credit Card'),
('TX-SIM-00042', '2022/09/05 08:29', '11968', '805756EB0', '1231', '8009B5440', 13232.59, 13232.59, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00043', '2022/09/05 11:21', '1217', '80060B3A0', '1768', '800DFB910', 8593.9, 8593.9, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00044', '2022/09/05 14:10', '11968', '805756EB0', '232814', '80FE82550', 1986.25, 1986.25, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00045', '2022/09/05 14:16', '1217', '80060B3A0', '498', '800469B70', 971295.01, 971295.01, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00046', '2022/09/05 14:35', '1217', '80060B3A0', '11', '800AECC10', 11375.95, 11375.95, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00047', '2022/09/05 14:38', '423', '8004943A0', '1120', '800DE3000', 13655.9, 13655.9, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00048', '2022/09/05 16:44', '423', '8004943A0', '13354', '801426610', 7959.76, 7959.76, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00049', '2022/09/05 18:59', '1217', '80060B3A0', '349', '80077B760', 1471.51, 1471.51, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00050', '2022/09/05 19:41', '1217', '80060B3A0', '1422', '80061FEF0', 1553.76, 1553.76, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00051', '2022/09/05 20:56', '21414', '800D10AF0', '11255', '800DF9310', 682.8, 682.8, 'US Dollar', 'US Dollar', 'Cheque'),
('TX-SIM-00052', '2022/09/06 01:59', '1595', '800774890', '8', '80009FC90', 15677.41, 15677.41, 'Canadian Dollar', 'Canadian Dollar', 'ACH'),
('TX-SIM-00053', '2022/09/06 02:21', '18326', '80DCF61B0', '1439', '811FC4DF0', 454.4, 454.4, 'US Dollar', 'US Dollar', 'Credit Card'),
('TX-SIM-00054', '2022/09/06 07:45', '1595', '800774890', '20', '8002BA9A0', 12694.85, 12694.85, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00055', '2022/09/06 10:15', '1595', '800774890', '531', '8003EA620', 16960.88, 16960.88, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00056', '2022/09/06 11:45', '11968', '805756EB0', '8814', '80AD106C0', 9525.94, 9525.94, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00057', '2022/09/06 15:03', '11968', '805756EB0', '5893', '802CCF530', 91725.03, 91725.03, 'Yuan', 'Yuan', 'ACH'),
('TX-SIM-00058', '2022/09/06 16:35', '11968', '805756EB0', '21316', '8084D0590', 3144.96, 3144.96, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00059', '2022/09/06 17:07', '11968', '805756EB0', '5383', '80224E670', 14695.36, 14695.36, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00060', '2022/09/06 23:46', '1110', '8075299E0', '118314', '80B84CB60', 3698.17, 3698.17, 'US Dollar', 'US Dollar', 'Cheque'),
('TX-SIM-00061', '2022/09/07 01:52', '1595', '800774890', '1291', '800822070', 614.29, 614.29, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00062', '2022/09/07 04:43', '11968', '805756EB0', '15640', '8050C86F0', 4835.59, 4835.59, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00063', '2022/09/07 10:46', '498', '800DAA850', '21876', '801C37770', 144.2, 144.2, 'Euro', 'Euro', 'Credit Card'),
('TX-SIM-00064', '2022/09/07 12:32', '1595', '800774890', '394', '8005150D0', 3714.47, 3714.47, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00065', '2022/09/07 14:40', '1595', '800774890', '1422', '8003CF520', 1223.16, 1223.16, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00066', '2022/09/07 16:05', '11968', '805756EB0', '24063', '80EA08F30', 17924.36, 17924.36, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00067', '2022/09/07 17:36', '11968', '805756EB0', '115332', '809713AD0', 16115.4, 16115.4, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00068', '2022/09/07 18:45', '11968', '805756EB0', '23041', '8061FBE10', 12249.89, 12249.89, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00069', '2022/09/08 02:34', '22129', '8070C98A0', '22129', '8070C98A0', 2179.84, 1860.27, 'Euro', 'US Dollar', 'ACH'),
('TX-SIM-00070', '2022/09/08 02:59', '119593', '8173C5700', '67359', '819B844C0', 110.67, 110.67, 'US Dollar', 'US Dollar', 'Cheque'),
('TX-SIM-00071', '2022/09/08 06:37', '11968', '805756EB0', '21260', '8010542C0', 19729.08, 19729.08, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00072', '2022/09/08 13:10', '1595', '800774890', '20', '8014FCB10', 13992.33, 13992.33, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00073', '2022/09/08 14:44', '165362', '81897DD10', '263574', '818D46070', 634.98, 634.98, 'Saudi Riyal', 'Saudi Riyal', 'Credit Card'),
('TX-SIM-00074', '2022/09/08 18:17', '11968', '805756EB0', '13577', '8054F7BB0', 92046.04, 92046.04, 'Yuan', 'Yuan', 'ACH'),
('TX-SIM-00075', '2022/09/08 18:58', '8', '80009B8E0', '29', '80C6EBF80', 71.95, 71.95, 'UK Pound', 'UK Pound', 'Credit Card'),
('TX-SIM-00076', '2022/09/08 20:22', '1595', '800774890', '1231', '800231EA0', 11767.05, 11767.05, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00077', '2022/09/09 01:42', '1291', '800245AD0', '21939', '800F36730', 6.45, 6.45, 'US Dollar', 'US Dollar', 'Credit Card'),
('TX-SIM-00078', '2022/09/09 02:29', '15516', '802B12070', '14652', '8116929A0', 6088.64, 6088.64, 'Euro', 'Euro', 'Cheque'),
('TX-SIM-00079', '2022/09/09 02:59', '1595', '800774890', '20', '80011E2A0', 2382.82, 2382.82, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00080', '2022/09/09 13:47', '1595', '800774890', '0', '800323610', 16277.32, 16277.32, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00081', '2022/09/09 16:39', '1595', '800774890', '14', '800278A90', 13155.94, 13155.94, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00082', '2022/09/09 20:34', '1595', '800774890', '12', '8001744F0', 5016.28, 5016.28, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00083', '2022/09/09 21:04', '1390', '801717B00', '15461', '802ED3AC0', 1184.6, 1184.6, 'Euro', 'Euro', 'Credit Card'),
('TX-SIM-00084', '2022/09/09 21:34', '1595', '800774890', '0', '8005F8370', 7707.74, 7707.74, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00085', '2022/09/10 01:11', '248739', '8128067C0', '11', '8132E5060', 275.21, 275.21, 'Brazil Real', 'Brazil Real', 'Cash'),
('TX-SIM-00086', '2022/09/10 17:54', '114948', '805D120A0', '19419', '8095E9C40', 4160.34, 4160.34, 'Euro', 'Euro', 'Cheque'),
('TX-SIM-00087', '2022/09/10 20:39', '27', '80BF623F0', '45104', '80F9F98C0', 12822.83, 12822.83, 'Australian Dollar', 'Australian Dollar', 'ACH'),
('TX-SIM-00088', '2022/09/10 21:17', '27', '80BF623F0', '16866', '80E119340', 16037.19, 16037.19, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00089', '2022/09/10 23:08', '27', '80BF623F0', '27', '8033AFA50', 19349700.95, 19349700.95, 'Yuan', 'Yuan', 'ACH'),
('TX-SIM-00090', '2022/09/12 11:16', '27', '80BF623F0', '22264', '801618FE0', 11290.88, 11290.88, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00091', '2022/09/12 13:05', '27', '80BF623F0', '33067', '80C58FC30', 1862.83, 1862.83, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00092', '2022/09/12 16:47', '27', '80BF623F0', '112646', '804C8AB80', 14290.87, 14290.87, 'Yuan', 'Yuan', 'ACH'),
('TX-SIM-00093', '2022/09/12 21:16', '27', '80BF623F0', '41335', '80F49DC80', 12421.63, 12421.63, 'Australian Dollar', 'Australian Dollar', 'ACH'),
('TX-SIM-00094', '2022/09/13 12:07', '27', '80BF623F0', '116703', '8064D3510', 15112.76, 15112.76, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00095', '2022/09/13 19:55', '27', '80BF623F0', '111913', '80B56A940', 13235.14, 13235.14, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00096', '2022/09/13 20:36', '27', '80BF623F0', '2692', '802DC7DC0', 7702.59, 7702.59, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00097', '2022/09/13 21:51', '27', '80BF623F0', '24202', '809B05D90', 1413455.12, 1413455.12, 'Rupee', 'Rupee', 'ACH'),
('TX-SIM-00098', '2022/09/14 09:36', '27', '80BF623F0', '27009', '80B880140', 12499.2, 12499.2, 'US Dollar', 'US Dollar', 'ACH'),
('TX-SIM-00099', '2022/09/14 10:06', '27', '80BF623F0', '16934', '805F5B360', 3902.74, 3902.74, 'Euro', 'Euro', 'ACH'),
('TX-SIM-00100', '2022/09/14 14:10', '27', '80BF623F0', '531', '8057B5070', 15565.88, 15565.88, 'US Dollar', 'US Dollar', 'ACH')
ON CONFLICT (tx_id) DO NOTHING;
