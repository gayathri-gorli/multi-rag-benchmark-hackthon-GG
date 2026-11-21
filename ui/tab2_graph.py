import math

import plotly.graph_objects as go
import streamlit as st

from core.graph_viz import communities, project, sphere_layout
from core.guardrails import PracticeArea
from ui.common import get_graph

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#72B7B2", "#E45756", "#9D755D", "#BAB0AC"]


def build_figure(kg, active_nodes: set, active_edges: set) -> go.Figure:
    G = kg.G
    pos, comm = sphere_layout(G), communities(G)
    fig = go.Figure()
    for hot, color, width in ((False, "rgba(150,150,150,0.35)", 2), (True, "#E8590C", 5)):
        xs, ys, zs = [], [], []
        for u, v, k in G.edges(keys=True):
            is_hot = (u, v, k) in active_edges or (not active_edges and u in active_nodes and v in active_nodes)
            if is_hot != hot:
                continue
            for n in (u, v):
                xs.append(pos[n][0]); ys.append(pos[n][1]); zs.append(pos[n][2])
            xs.append(None); ys.append(None); zs.append(None)
        if xs:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width),
                                       hoverinfo="skip", showlegend=False))
    nodes = list(G.nodes)
    sel = [n in active_nodes for n in nodes]
    fig.add_trace(go.Scatter3d(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], z=[pos[n][2] for n in nodes],
        mode="markers+text", text=[kg.label(n) if s else "" for n, s in zip(nodes, sel)], textposition="top center",
        marker=dict(size=[12 if s else 6 for s in sel], color=[PALETTE[comm[n] % len(PALETTE)] for n in nodes],
                    line=dict(color=["#E8590C" if s else "rgba(0,0,0,0)" for s in sel], width=4)),
        hovertext=[f"{kg.label(n)}<br>degree {G.degree(n)}<br>cluster {comm[n]}" for n in nodes],
        hoverinfo="text", showlegend=False))
    axis = dict(visible=False)
    fig.update_layout(scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube"),
                      margin=dict(l=0, r=0, t=0, b=0), height=620)
    return fig


def build_figure_2d(kg, active_nodes: set, active_edges: set, azimuth: float, elevation: float) -> go.Figure:
    """Rotatable sphere drawn as an SVG scatter (no WebGL). Far-side nodes/edges are dimmed and smaller."""
    G = kg.G
    pr, comm = project(sphere_layout(G), azimuth, elevation), communities(G)
    fig = go.Figure()
    t = [i / 100 * 2 * math.pi for i in range(101)]
    fig.add_trace(go.Scatter(x=[math.cos(a) for a in t], y=[math.sin(a) for a in t], mode="lines",
                             line=dict(color="rgba(120,120,120,0.45)", width=1), hoverinfo="skip", showlegend=False))
    for kind, color, width in (("back", "rgba(150,150,150,0.15)", 1), ("front", "rgba(110,110,110,0.45)", 1.5),
                               ("hot", "#E8590C", 4)):
        xs, ys = [], []
        for u, v, k in G.edges(keys=True):
            hot = (u, v, k) in active_edges or (not active_edges and u in active_nodes and v in active_nodes)
            front = pr[u][2] > 0 and pr[v][2] > 0
            cat = "hot" if hot else "front" if front else "back"
            if cat != kind:
                continue
            xs += [pr[u][0], pr[v][0], None]
            ys += [pr[u][1], pr[v][1], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width),
                                     hoverinfo="skip", showlegend=False))
    nodes = sorted(G.nodes, key=lambda n: pr[n][2])  # draw far nodes first so near ones sit on top
    sel = [n in active_nodes for n in nodes]
    front = [pr[n][2] > 0 for n in nodes]
    fig.add_trace(go.Scatter(
        x=[pr[n][0] for n in nodes], y=[pr[n][1] for n in nodes], mode="markers+text",
        text=[kg.label(n) if s and f else "" for n, s, f in zip(nodes, sel, front)], textposition="top center",
        marker=dict(size=[16 if s else 11 if f else 7 for s, f in zip(sel, front)],
                    color=[PALETTE[comm[n] % len(PALETTE)] for n in nodes],
                    opacity=[1.0 if f else 0.3 for f in front],
                    line=dict(color=["#E8590C" if s else "rgba(0,0,0,0)" for s in sel], width=3)),
        hovertext=[f"{kg.label(n)}<br>degree {G.degree(n)}<br>cluster {comm[n]}"
                   f"{'' if f else '<br>(far side)'}" for n, f in zip(nodes, front)],
        hoverinfo="text", showlegend=False))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=620, plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False, range=[-1.25, 1.25]),
                      yaxis=dict(visible=False, range=[-1.25, 1.25], scaleanchor="x", scaleratio=1))
    return fig


def render():
    st.subheader("Interactive Spherical Knowledge Graph Canvas")
    area = st.selectbox("Practice Area", [a.value for a in PracticeArea], key="kg_area")
    kg = get_graph(area)
    if kg.G.number_of_nodes() == 0:
        st.info("No knowledge graph yet. Ingest documents in Tab 1 with graph extraction enabled.")
        return
    labels = {kg.label(n) + f"  ·  {n}": n for n in sorted(kg.G.nodes, key=kg.label)}
    edge_labels = {f"{kg.label(u)} —{r}→ {kg.label(v)}": (u, v, k) for u, v, k, r in kg.edge_list()}
    cur_nodes = st.session_state.active_entities.get(area, set())
    cur_edges = st.session_state.active_edges.get(area, set())

    c1, c2 = st.columns(2)
    sel_nodes = c1.multiselect("Select entities (active retrieval filter)", list(labels),
                               default=[l for l, n in labels.items() if n in cur_nodes])
    sel_edges = c2.multiselect("Select relations", list(edge_labels),
                               default=[l for l, e in edge_labels.items() if e in cur_edges])
    nodes, edges = {labels[l] for l in sel_nodes}, {edge_labels[l] for l in sel_edges}
    st.session_state.active_entities[area], st.session_state.active_edges[area] = nodes, edges

    if nodes or edges:
        st.caption(f"Graph RAG is restricted to {len(nodes) or 'all'} entities and {len(edges) or 'all'} relations "
                   "for this practice area. Clear both selections to remove the filter.")
    view = st.radio("View", ["Sphere projection (no WebGL needed)", "True 3D (needs WebGL)"], horizontal=True)
    if view.startswith("Sphere"):
        v1, v2 = st.columns(2)
        az = v1.slider("Rotate (azimuth °)", 0, 360, 30, key="kg_az")
        el = v2.slider("Tilt (elevation °)", -90, 90, 20, key="kg_el")
        st.plotly_chart(build_figure_2d(kg, nodes, edges, az, el), width="stretch")
        st.caption("Use the sliders to rotate the sphere; dimmed points are on the far side. Colour = community "
                   "cluster; orange = selected. Select entities and relations with the pickers above.")
    else:
        st.plotly_chart(build_figure(kg, nodes, edges), width="stretch")
        st.caption("Drag to rotate. If the chart is blank or shows a WebGL error, switch to the sphere projection.")
