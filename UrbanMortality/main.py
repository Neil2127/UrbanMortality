#!/usr/bin/env python3
"""
UrbanMortality — Wildlife mortality risk mapper
Location : Chicago, IL — Mississippi Flyway corridor
           (Documented #1 US city for fatal bird-glass collisions;
            McCormick Place alone kills ~1,000 birds/year; flyway
            concentrates millions of migrants along Lake Michigan shore)

Datasets used
  • iNaturalist  – citizen-science "dead" wildlife reports  (free API, no key)
  • OpenStreetMap – building footprints + drive-network  (via OSMnx)
  • Lake Michigan shoreline – proxy for Mississippi Flyway migration pressure

Pipeline
  1. Create 300-m study grid over Chicago downtown
  2. Fetch bird / mammal mortality incidents from iNaturalist
  3. Fetch OSM building footprints (glass-type tag) + road network
  4. Compute per-cell features  (building density, glass fraction,
                                  road segments, shore distance, latitude)
  5. Train Random Forest  (labels = cells with ≥1 mortality incident)
  6. Assign risk tier  (Low / Medium / High)  via quantile thresholds
  7. Generate mortality risk map + intervention recommendations → PNG + CSV
"""

import warnings
warnings.filterwarnings("ignore")

import importlib.metadata as _imm
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import requests
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Config ──────────────────────────────────────────────────────────────────────
SOUTH, WEST, NORTH, EAST = 41.830, -87.680, 41.920, -87.600   # Chicago downtown
BBOX  = (SOUTH, WEST, NORTH, EAST)
GRID_DEG  = 0.003       # ≈ 300 m per cell  (city-block tier)
UTM       = "EPSG:32616"  # UTM Zone 16N — metric distances for Chicago
LAKE_LNG  = -87.523     # Lake Michigan west-shore longitude (migration corridor)

# Building tags indicating significant glass facade (commercial / civic stock)
GLASS_TYPES = {"commercial", "office", "retail", "hotel", "civic",
               "public", "mixed_use", "yes"}

# ── osmnx version shim (1.x vs 2.x API break) ───────────────────────────────────
_oxv = tuple(int(x) for x in _imm.version("osmnx").split(".")[:2])

def _osm_features(tags):
    if _oxv >= (2, 0):
        return ox.features_from_bbox((WEST, SOUTH, EAST, NORTH), tags)
    return ox.features_from_bbox(north=NORTH, south=SOUTH, east=EAST, west=WEST, tags=tags)

def _osm_graph():
    if _oxv >= (2, 0):
        return ox.graph_from_bbox((WEST, SOUTH, EAST, NORTH), network_type="drive")
    return ox.graph_from_bbox(north=NORTH, south=SOUTH, east=EAST, west=WEST, network_type="drive")


# ── 1. Study grid ───────────────────────────────────────────────────────────────
def make_grid():
    cells = [
        box(x, y, x + GRID_DEG, y + GRID_DEG)
        for x in np.arange(WEST,  EAST,  GRID_DEG)
        for y in np.arange(SOUTH, NORTH, GRID_DEG)
    ]
    gdf = gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs="EPSG:4326")
    gdf["cx"]         = gdf.geometry.centroid.x
    gdf["cy"]         = gdf.geometry.centroid.y
    gdf["shore_dist"] = (gdf["cx"] - LAKE_LNG).abs()   # distance from migration corridor
    return gdf


# ── 2. iNaturalist mortality records ────────────────────────────────────────────
# term_id=17 → "Alive or Dead";  term_value_id=19 → "Dead"
def fetch_mortality():
    records = []
    for taxon_id, kind in [(3, "bird"), (40151, "mammal")]:
        for page in range(1, 6):           # up to 5 pages × 200 = 1 000 per taxon
            try:
                r = requests.get(
                    "https://api.inaturalist.org/v1/observations",
                    timeout=30,
                    params={
                        "taxon_id": taxon_id,
                        "swlat": SOUTH, "swlng": WEST, "nelat": NORTH, "nelng": EAST,
                        "quality_grade": "research,needs_id",
                        "term_id": 17, "term_value_id": 19,
                        "per_page": 200, "page": page,
                    },
                )
            except Exception as exc:
                print(f"  [warn] iNaturalist request failed: {exc}")
                break
            if r.status_code != 200:
                break
            results = r.json().get("results", [])
            for obs in results:
                loc = obs.get("location")
                if loc:
                    la, lo = map(float, loc.split(","))
                    records.append({"lat": la, "lng": lo, "kind": kind})
            if len(results) < 200:
                break

    if len(records) < 10:
        print("  [note] Sparse iNaturalist data — using realistic synthetic incidents.")
        rng = np.random.default_rng(42)
        # Concentrate mortalities in known Chicago collision hotspots:
        # The Loop (glass towers), Streeterville, Grant Park lakefront
        lats = np.concatenate([
            rng.normal(41.878, 0.010, 90),   # The Loop / Streeterville
            rng.normal(41.857, 0.008, 30),   # South Loop / Museum Campus
            rng.normal(41.905, 0.008, 20),   # Lincoln Park lakefront
        ])
        lngs = np.concatenate([
            rng.normal(-87.630, 0.014, 90),
            rng.normal(-87.618, 0.010, 30),
            rng.normal(-87.635, 0.010, 20),
        ])
        records = [{"lat": float(la), "lng": float(lo), "kind": "bird"}
                   for la, lo in zip(lats, lngs)]
        records += [{"lat": float(la), "lng": float(lo), "kind": "mammal"}
                    for la, lo in zip(rng.normal(41.868, 0.012, 25),
                                      rng.normal(-87.655, 0.012, 25))]

    df = pd.DataFrame(records)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat), crs="EPSG:4326")


# ── 3. OSM buildings + road network ─────────────────────────────────────────────
def fetch_osm():
    print("  buildings … ", end="", flush=True)
    raw   = _osm_features({"building": True})
    bldgs = raw[raw.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    bldgs["glass"] = bldgs["building"].astype(str).isin(GLASS_TYPES).astype(int)
    print(len(bldgs))

    print("  road network … ", end="", flush=True)
    G     = _osm_graph()
    edges = ox.graph_to_gdfs(G, nodes=False)
    print(len(edges), "segments")

    return bldgs[["geometry", "glass"]], edges[["geometry"]]


# ── 4. Spatial features per grid cell ───────────────────────────────────────────
def compute_features(grid, bldgs, roads, mort):
    g = grid.copy().to_crs(UTM)
    b = bldgs.to_crs(UTM)
    r = roads.to_crs(UTM)
    m = mort.to_crs(UTM)

    # Building count + glass fraction
    bj = gpd.sjoin(b, g[["cell_id", "geometry"]], how="left", predicate="intersects")
    ba = bj.groupby("cell_id").agg(bldg_n=("glass", "count"),
                                    glass_n=("glass", "sum")).reset_index()
    g = g.merge(ba, on="cell_id", how="left").fillna(0)
    g["glass_frac"] = g["glass_n"] / (g["bldg_n"] + 1)

    # Road segment count (proxy for road mortality exposure)
    rj = gpd.sjoin(r, g[["cell_id", "geometry"]], how="left", predicate="intersects")
    ra = rj.groupby("cell_id").size().rename("road_segs").reset_index()
    g  = g.merge(ra, on="cell_id", how="left").fillna(0)

    # Observed mortality incident count (training labels)
    mj = gpd.sjoin(m[["geometry", "kind"]], g[["cell_id", "geometry"]],
                   how="left", predicate="within")
    ma = mj.groupby("cell_id").size().rename("mort_n").reset_index()
    g  = g.merge(ma, on="cell_id", how="left").fillna(0)

    return g.to_crs("EPSG:4326")


# ── 5. Random Forest classifier ──────────────────────────────────────────────────
FEAT = ["bldg_n", "glass_frac", "road_segs", "shore_dist", "cy"]

def train_rf(grid):
    X = grid[FEAT].fillna(0).values
    y = (grid["mort_n"] >= 1).astype(int).values

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        class_weight="balanced", random_state=42,
    )
    clf.fit(X, y)

    grid       = grid.copy()
    grid["risk_p"] = clf.predict_proba(X)[:, 1]

    # Quantile thresholds → balanced Low / Medium / High tiers
    q1 = grid["risk_p"].quantile(0.60)
    q2 = grid["risk_p"].quantile(0.85)
    grid["risk"] = pd.cut(
        grid["risk_p"],
        bins=[-0.001, q1, q2, 1.001],
        labels=["Low", "Medium", "High"],
    )
    return grid, clf


# ── 6. Counterfactual intervention simulation ─────────────────────────────────
INTERVENTION_SCENARIOS = {
    "Anti-collision window film": {
        "glass_frac": lambda s: s * 0.30,
    },
    "Reduced night lighting": {
        "road_segs": lambda s: np.maximum(s * 0.80, 0),
        "glass_frac": lambda s: s * 0.90,
    },
    "Vegetation buffer": {
        "glass_frac": lambda s: s * 0.70,
        "road_segs": lambda s: np.maximum(s * 0.90, 0),
    },
    "Wildlife crossing corridor": {
        "road_segs": lambda s: np.maximum(s * 0.50, 0),
    },
}


def _slug(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def simulate_interventions(grid, clf):
    grid = grid.copy()
    grid["baseline_risk"] = grid["risk_p"].astype(float)

    scenario_names = list(INTERVENTION_SCENARIOS)
    slugs = [_slug(name) for name in scenario_names]
    risk_cols = []
    reduction_cols = []

    for name, slug in zip(scenario_names, slugs):
        scenario = grid.copy()
        transforms = INTERVENTION_SCENARIOS[name]
        for field, transform in transforms.items():
            scenario[field] = transform(scenario[field])

        col_risk = f"risk_{slug}"
        col_reduction = f"reduction_{slug}"
        scenario_risk = clf.predict_proba(scenario[FEAT].fillna(0).values)[:, 1]

        grid[col_risk] = scenario_risk
        grid[col_reduction] = np.where(
            grid["baseline_risk"] > 0,
            100.0 * (grid["baseline_risk"] - scenario_risk) / grid["baseline_risk"],
            0.0,
        )
        risk_cols.append(col_risk)
        reduction_cols.append(col_reduction)

    reductions = grid[reduction_cols].to_numpy(dtype=float)
    best_idx = np.nanargmax(reductions, axis=1)

    grid["best_intervention"] = [scenario_names[i] for i in best_idx]
    grid["expected_reduction"] = reductions[np.arange(len(grid)), best_idx]
    grid["intervention_risk"] = grid[risk_cols].to_numpy(dtype=float)[np.arange(len(grid)), best_idx]

    grid["intervention_rankings"] = [
        "; ".join(
            f"{name} ({reduction:.1f}%)"
            for name, reduction in sorted(
                zip(scenario_names, row), key=lambda item: item[1], reverse=True
            )
        )
        for row in reductions
    ]

    # Legacy alias for compatibility with older output consumers.
    grid["intervention"] = grid["best_intervention"]
    return grid


# ── 7. Output map ────────────────────────────────────────────────────────────────
RISK_PAL = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
INT_PAL  = {
    "Anti-collision window film":                "#c0392b",
    "Reduced night lighting":                    "#9b59b6",
    "Vegetation buffer":                         "#d35400",
    "Wildlife crossing corridor":                "#8e44ad",
}

def plot_map(grid, mort):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # ── Left: risk map ──
    grid["_rc"] = grid["risk"].astype(str).map(RISK_PAL).fillna("#bbb")
    grid.plot(color=grid["_rc"], ax=axes[0], alpha=0.75, edgecolor="white", linewidth=0.2)

    for kind, color, marker, label in [
        ("bird",   "#1a237e", "o", "Bird mortality"),
        ("mammal", "#6a0dad", "^", "Mammal mortality"),
    ]:
        sub = mort[mort["kind"] == kind]
        if len(sub):
            sub.plot(ax=axes[0], color=color, markersize=5,
                     marker=marker, alpha=0.7, zorder=5)

    axes[0].legend(
        handles=[Patch(fc=c, label=l) for l, c in RISK_PAL.items()] + [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a237e",
                       markersize=7, label="Bird mortality"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#6a0dad",
                       markersize=7, label="Mammal mortality"),
        ],
        fontsize=8, loc="upper right", framealpha=0.85,
    )
    axes[0].set_title("Wildlife Mortality Risk — Chicago, IL", fontsize=12, fontweight="bold")
    axes[0].set_axis_off()

    # ── Right: intervention map ──
    grid["_ic"] = grid["best_intervention"].map(INT_PAL).fillna("#bbb")
    grid.plot(color=grid["_ic"], ax=axes[1], alpha=0.75, edgecolor="white", linewidth=0.2)
    axes[1].legend(
        handles=[Patch(fc=c, label=l) for l, c in INT_PAL.items()],
        fontsize=7.5, loc="upper right", framealpha=0.85,
    )
    axes[1].set_title("Intervention Recommendations", fontsize=12, fontweight="bold")
    axes[1].set_axis_off()

    fig.suptitle(
        "UrbanMortality | Chicago, IL — Mississippi Flyway\n"
        "Bird–Glass Building Collisions & Urban Road Mortality Risk",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig("mortality_risk_map.png", dpi=150, bbox_inches="tight")
    print("  ✓  mortality_risk_map.png")


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  UrbanMortality  |  Chicago, IL — Mississippi Flyway")
    print(f"  osmnx {_imm.version('osmnx')}  |  grid {GRID_DEG}° ≈ 300 m")
    print("=" * 60)

    print("\n[1/5] Building study grid …")
    grid = make_grid()
    print(f"      {len(grid)} cells")

    print("\n[2/5] Fetching iNaturalist mortality records …")
    mort = fetch_mortality()
    print(f"      birds={( mort['kind'] == 'bird').sum()}  "
          f"mammals={(mort['kind'] == 'mammal').sum()}")

    print("\n[3/5] Fetching OSM buildings + roads …")
    bldgs, roads = fetch_osm()

    print("\n[4/5] Computing spatial features …")
    grid = compute_features(grid, bldgs, roads, mort)

    print("\n[5/5] Training Random Forest …")
    grid, clf = train_rf(grid)
    grid = simulate_interventions(grid, clf)

    # ── Summary ──
    rc = grid["risk"].value_counts().to_dict()
    print(f"\n  Risk distribution:  High={rc.get('High', 0)}  "
          f"Medium={rc.get('Medium', 0)}  Low={rc.get('Low', 0)}")

    fi = dict(zip(FEAT, clf.feature_importances_))
    print("\n  Feature importances:")
    for f, v in sorted(fi.items(), key=lambda x: -x[1]):
        print(f"    {f:<14s}  {v:.3f}")

    top = (grid[grid["risk"] == "High"]
           .sort_values("risk_p", ascending=False)
           .head(5))
    print("\n  Top-5 high-risk cells:")
    for _, row in top.iterrows():
        print(f"    ({row.cy:.4f}°N, {abs(row.cx):.4f}°W)  "
              f"p={row.risk_p:.2f}  glass={row.glass_frac:.2f}  "
              f"→ {row['intervention']}")

    # ── Save ──
    save_cols = [
        "cell_id", "cy", "cx", "bldg_n", "glass_frac", "road_segs",
        "shore_dist", "mort_n", "risk_p", "risk", "baseline_risk",
        "intervention_risk", "expected_reduction", "best_intervention",
        "intervention_rankings",
    ]
    grid[save_cols].to_csv("mortality_risk.csv", index=False)
    print("\n  ✓  mortality_risk.csv")
    import os
    os.makedirs("cache", exist_ok=True)
    cache_grid = grid.copy()
    cache_grid["risk"] = cache_grid["risk"].astype(str)
    cache_grid[save_cols + ["geometry"]].to_file("cache/grid.geojson", driver="GeoJSON")
    mort[["geometry", "kind"]].to_file("cache/incidents.geojson", driver="GeoJSON")
    print("  ✓  cache/grid.geojson + cache/incidents.geojson")
    plot_map(grid, mort)
    print("\n  Done.")


if __name__ == "__main__":
    main()
