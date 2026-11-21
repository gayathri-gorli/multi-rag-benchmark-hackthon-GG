"""Tab 2 helpers: community detection and a spherical layout (Fibonacci sphere, communities
laid out contiguously so clusters appear as caps on the sphere). No UI imports."""
import math

import networkx as nx


def communities(G) -> dict[str, int]:
    if G.number_of_nodes() == 0:
        return {}
    comms = nx.community.greedy_modularity_communities(nx.Graph(G))
    return {n: i for i, c in enumerate(sorted(comms, key=lambda c: (-len(c), sorted(c)[0]))) for n in c}


def sphere_layout(G, radius: float = 1.0) -> dict[str, tuple[float, float, float]]:
    comm = communities(G)
    order = sorted(G.nodes, key=lambda n: (comm.get(n, 0), n))
    n = len(order)
    golden = math.pi * (3 - math.sqrt(5))
    pos = {}
    for i, node in enumerate(order):
        y = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1 - y * y))
        phi = i * golden
        pos[node] = (radius * r * math.cos(phi), radius * y, radius * r * math.sin(phi))
    return pos


def project(pos: dict[str, tuple[float, float, float]], azimuth_deg: float = 30.0,
            elevation_deg: float = 20.0) -> dict[str, tuple[float, float, float]]:
    """Rotate the sphere and project orthographically to the screen. Returns node -> (x, y, depth);
    depth > 0 means the node is on the hemisphere facing the viewer. Pure maths, so the Tab 2 view
    needs no WebGL."""
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    out = {}
    for n, (x, y, z) in pos.items():
        x1, z1 = x * math.cos(a) + z * math.sin(a), -x * math.sin(a) + z * math.cos(a)
        y2, z2 = y * math.cos(e) - z1 * math.sin(e), y * math.sin(e) + z1 * math.cos(e)
        out[n] = (x1, y2, z2)
    return out
