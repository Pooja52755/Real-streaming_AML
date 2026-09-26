"""
Graph Visualization Module using Plotly & NetworkX (Root Module)
Renders interactive graph network topology for AML transaction analysis.
Matches Screenshot 4: Star network with Sender (★) in center and Hop-1 Receivers (●) on perimeter.
"""

import plotly.graph_objects as go
import networkx as nx
import fraud_data


def _build_hover_text(node, data):
    """Build a rich hover tooltip for a given node matching the dashboard screenshot design."""
    hop = data.get("hop", 0)

    if hop == 0:
        # Sender Node
        entity_name = data.get("entity_name") or node
        bank_name = data.get("bank_name", "Global Bank")
        bank_id = data.get("bank_id", "BNK-001")
        entity_id = data.get("entity_id", f"ENT-{node[:8]}")
        gat_signal = data.get("gat_signal", "HIGH")
        risk_score = data.get("risk_score", 98)

        return (
            f"<b>🔴 SENDER ACCOUNT</b><br>"
            f"<b>{entity_name}</b><br>"
            f"Account: {node}<br>"
            f"Entity ID: {entity_id}<br>"
            f"Bank: {bank_name} (ID: {bank_id})<br>"
            f"GAT Signal: {gat_signal}<br>"
            f"GAT Risk Score: {risk_score}/100"
        )
    else:
        # Hop-1 Receiver Node
        entity_name = data.get("entity_name") or node
        bank_name = data.get("bank_name", "Global Bank")
        bank_id = data.get("bank_id", "BNK-001")
        entity_id = data.get("entity_id", f"ENT-{node[:8]}")
        amount = data.get("amount", "Transfer")

        return (
            f"<b>🟡 HOP-1 RECEIVER</b><br>"
            f"<b>{entity_name}</b><br>"
            f"Account: {node}<br>"
            f"Amount Received: {amount}<br>"
            f"Entity ID: {entity_id}<br>"
            f"Bank: {bank_name} (ID: {bank_id})"
        )


def render_plotly_graph(tx_id_or_group_id, include_2hop=False):
    """
    Renders an interactive Plotly figure representing the transaction graph.
    Shows source (★) → hop-1 receivers (●) with actual transfer amounts.
    Hovering on any node shows the real customer & bank profile.
    """
    G = fraud_data.create_network_graph(tx_id_or_group_id, include_2hop=include_2hop)

    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No graph data available",
            paper_bgcolor="#f8fafc",
            plot_bgcolor="#f8fafc",
            height=520,
        )
        return fig

    pos = nx.spring_layout(G, seed=42, k=2.5)

    edge_traces = []
    annotation_list = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        hop = edge[2].get("hop", 1)
        amount = edge[2].get("amount", "Transfer")

        color = "#ef4444" if hop == 1 else "#cbd5e1"
        width = 2.5 if hop == 1 else 1.0

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="text",
            hovertext=f"<b>Transfer:</b> {edge[0]} ➔ {edge[1]}<br><b>Amount:</b> {amount}",
            showlegend=False,
        ))

        # Mid-point label for transfer amount
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        annotation_list.append(dict(
            x=mx, y=my,
            text=f"<b>{amount}</b>",
            showarrow=False,
            font=dict(size=9, color="#dc2626"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#fecaca",
            borderwidth=1,
            borderpad=2,
        ))

    # Node traces
    node_groups = {"source": [], "hop1": []}
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        hop = data.get("hop", 1)
        key = "source" if hop == 0 else "hop1"
        node_groups[key].append((node, data, x, y))

    node_traces = []

    group_styles = {
        "source": dict(
            name="Sender Account",
            symbol="star",
            size=36,
            color="#ef4444",
            line_color="#ffffff",
            line_width=3,
            text_color="#1e293b",
        ),
        "hop1": dict(
            name="Hop-1 Receivers",
            symbol="circle",
            size=26,
            color="#f59e0b",
            line_color="#ffffff",
            line_width=2,
            text_color="#334155",
        ),
    }

    for key, nodes in node_groups.items():
        if not nodes:
            continue
        style = group_styles[key]
        xs = [n[2] for n in nodes]
        ys = [n[3] for n in nodes]
        labels = [n[0] for n in nodes]
        hover = [_build_hover_text(n[0], n[1]) for n in nodes]

        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            name=style["name"],
            text=labels,
            textposition="top center",
            textfont=dict(size=9, color=style["text_color"]),
            hoverinfo="text",
            hovertext=hover,
            hoverlabel=dict(
                bgcolor="#0f172a",
                font_size=11,
                font_color="white",
                bordercolor="#334155",
            ),
            marker=dict(
                symbol=style["symbol"],
                size=style["size"],
                color=style["color"],
                line=dict(width=style["line_width"], color=style["line_color"]),
                opacity=0.95,
            ),
            showlegend=True,
        ))

    all_traces = edge_traces + node_traces

    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            showlegend=True,
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                font=dict(size=11),
            ),
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            annotations=annotation_list,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#f8fafc",
            plot_bgcolor="#f8fafc",
            height=540,
            title=dict(
                text=f"<b>Transaction Network — {tx_id_or_group_id}</b>   "
                     f"<span style='font-size:12px;color:#64748b;'>"
                     f"★ Sender  🟡 Hop-1 Receivers  — Hover any node for customer profile</span>",
                font=dict(size=13, color="#1e293b"),
                x=0.0,
            ),
        )
    )

    return fig
