"""Shared plotting constants so colours/labels/order stay consistent across all app pages."""

# Canonical policy order — the SAME order in every cost bar chart / legend.
POLICY_ORDER = [
    "Backward Induction",
    "DP-Heuristic",
    "Price-Oriented",
    "Night Charging",
    "Always-Maximum",
    "Minimum-Charge",
    "Always-Minimum",
]

# Canonical policy → colour map (single source of truth; same colour everywhere).
# Keys follow POLICY_ORDER; each policy keeps its colour across all figures.
POLICY_COLORS = {
    "Backward Induction": "#4477AA",
    "DP-Heuristic":       "#009988",
    "Price-Oriented":     "#EE6677",
    "Night Charging":     "#AA3377",
    "Always-Maximum":   "#228833",
    "Minimum-Charge":     "#EE7733",
    "Always-Minimum":     "#BBBBBB",
}
