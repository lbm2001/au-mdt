"""Shared plotting constants — canonical colours, labels, and order for all plots."""

_NAMED_RGB = {"orange": "255,165,0", "lightgray": "211,211,211"}


def rgba(color: str, alpha: float) -> str:
    """rgba() string for a hex (#RRGGBB) or named colour, at the given opacity."""
    if color.startswith("#"):
        h = color.lstrip("#")
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"
    return f"rgba({_NAMED_RGB.get(color, '128,128,128')},{alpha})"


POLICY_ORDER = [
    "Backward Induction",
    "Battery Level Urgency",
    "Departure Urgency",
    "Price-Oriented",
    "Night Charging",
    "Always-Maximum",
    "Minimum Battery Level",
    "Always-Minimum",
]

POLICY_COLORS = {
    "Backward Induction":       "#4477AA",
    "Battery Level Urgency":    "#009988",
    "Departure Urgency":        "#3399BB",
    "Price-Oriented":           "#EE6677",
    "Night Charging":           "#AA3377",
    "Always-Maximum":           "#228833",
    "Minimum Battery Level":    "#EE7733",
    "Always-Minimum":           "#BBBBBB",
}

MODEL_COLORS = {
    "Baseline":                        "#4477AA",
    "Negative Binomial (fixed k)":     "#EE6677",
    "Negative Binomial (Poisson k)":   "#228833",
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
