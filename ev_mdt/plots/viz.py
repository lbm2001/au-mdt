"""Shared plotting constants — canonical colours, labels, and order for all plots."""

POLICY_ORDER = [
    "Backward Induction",
    "DP-Heuristic",
    "Price-Oriented",
    "Night Charging",
    "Always-Maximum",
    "Minimum Battery Level",
    "Always-Minimum",
]

POLICY_COLORS = {
    "Backward Induction":   "#4477AA",
    "DP-Heuristic":         "#009988",
    "Price-Oriented":       "#EE6677",
    "Night Charging":       "#AA3377",
    "Always-Maximum":       "#228833",
    "Minimum Battery Level": "#EE7733",
    "Always-Minimum":       "#BBBBBB",
}

SWEEP_PALETTE = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#BBBBBB", "#882255", "#44AA99", "#999933", "#DDCC77",
]

SWEEP_AXIS_LABEL = {
    "pricing_model":     "Pricing model",
    "pricing_season":    "Season",
    "pricing_daytype":   "Day type",
    "pricing_crisis":    "Energy-crisis data",
    "penalty":           "Penalty (€/h)",
    "beta":              "Discount factor β",
    "horizon":           "Horizon T (h)",
    "departure_profile": "Departure profile",
    "mobility_model":    "Mobility model",
}
