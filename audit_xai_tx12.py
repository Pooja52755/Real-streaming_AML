import torch
import numpy as np
from fraud_data import RealTimeStreamingEngine, NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES

engine = RealTimeStreamingEngine()
engine.reset_state()

for _ in range(12):
    tx = engine.process_next_transaction()

# Step 12 is Tx #12
print("=== AUDIT FOR TRANSACTION #12 ===")
print(f"Tx ID: {tx['tx_id']}")
print(f"From: {tx['from_account']} -> To: {tx['to_account']}")
print(f"1. Original GAT Raw Logit: {tx['gat_raw_logit']}")
print(f"2. GAT Probability: {tx['gat_prob']}")

from_acc = tx['from_account']
to_acc = tx['to_account']
neighbor_nodes = set([from_acc, to_acc])
for u in [from_acc, to_acc]:
    if engine.G.has_node(u):
        neighbor_nodes.update(engine.G.successors(u))
        neighbor_nodes.update(engine.G.predecessors(u))
node_order = [from_acc, to_acc] + [n for n in neighbor_nodes if n not in [from_acc, to_acc]]
node_to_idx = {n: i for i, n in enumerate(node_order)}
node_feats_raw = np.array([engine._compute_node_feature_vec(n) for n in node_order])
scaled_nodes = engine.node_scaler.transform(node_feats_raw)

src_edges, dst_edges, edge_feats = [], [], []
for u in node_order:
    if engine.G.has_node(u):
        for v in engine.G.successors(u):
            if v in node_to_idx:
                for k, edata in engine.G[u][v].items():
                    src_edges.append(node_to_idx[u])
                    dst_edges.append(node_to_idx[v])
                    edge_feats.append(edata.get('edge_raw'))

scaled_edges = engine.edge_scaler.transform(edge_feats)
scaled_target_edge = engine.edge_scaler.transform([edge_feats[-1]])

x_tensor = torch.tensor(scaled_nodes, dtype=torch.float32)
msg_edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long)
msg_edge_attr = torch.tensor(scaled_edges, dtype=torch.float32)
target_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
target_edge_attr = torch.tensor(scaled_target_edge, dtype=torch.float32)

x_in = x_tensor.clone().detach().requires_grad_(True)
target_edge_in = target_edge_attr.clone().detach().requires_grad_(True)

out_logit = engine.model(x_in, msg_edge_index, msg_edge_attr, target_edge_index, target_edge_in)
out_logit.backward()

node_attr = (x_in.grad[0] * x_in[0]).abs().detach().cpu().numpy()
edge_attr = (target_edge_in.grad[0] * target_edge_in[0]).abs().detach().cpu().numpy()

print("\n3. EVERY FEATURE GIVEN TO XAI METHOD (Sender Node [13] + Target Edge [20]):")
header = f"{'Type':<6} , {'Idx':<3} , {'Feature Name':<45} , {'Raw Input':<12} , {'Scaled x':<12} , {'dL/dx':<12} , {'Attribution':<15}"
print(header)
print("-" * len(header))

all_drivers = []
for idx, ((k, label), raw_v, scaled_v, grad_v, attr_v) in enumerate(zip(NODE_FEATURE_NAMES, node_feats_raw[0], scaled_nodes[0], x_in.grad[0].numpy(), node_attr)):
    all_drivers.append((label, attr_v))
    print(f"NODE   , {idx:<3} , {label:<45} , {raw_v:<12.4f} , {scaled_v:<12.4f} , {grad_v:<12.4f} , {attr_v:<15.6f}")

for idx, ((k, label), raw_v, scaled_v, grad_v, attr_v) in enumerate(zip(EDGE_FEATURE_NAMES, edge_feats[-1], scaled_target_edge[0], target_edge_in.grad[0].numpy(), edge_attr)):
    all_drivers.append((label, attr_v))
    print(f"EDGE   , {idx:<3} , {label:<45} , {raw_v:<12.4f} , {scaled_v:<12.4f} , {grad_v:<12.4f} , {attr_v:<15.6f}")

all_drivers.sort(key=lambda x: x[1], reverse=True)
top3 = all_drivers[:3]
tot = sum([d[1] for d in top3]) + 1e-6

print("\n4. TOP 3 ATTRIBUTIONS & NORMALIZED PERCENTAGES DISPLAYED IN UI:")
for label, val in top3:
    pct = (val / tot) * 100.0
    print(f"  * {label}: Attribution = {val:.6f} -> UI Display = +{pct:.1f}% relative GAT attribution")
