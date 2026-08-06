"""Canonical ROOT tree names and ordered compatibility fallbacks."""

R_EVENTS = "rEvents"
R_PARTICLES = "rParticles"
G_EVENTS = "gEvents"
S_EVENTS = "sEvents"
S_PARTICLES = "sParticles"

_ALIASES = {
    R_EVENTS: (R_EVENTS, "ReconstructedEvents"),
    "ReconstructedEvents": ("ReconstructedEvents", R_EVENTS),
    R_PARTICLES: (R_PARTICLES, "ReconstructedParticles", "Events"),
    "ReconstructedParticles": ("ReconstructedParticles", R_PARTICLES, "Events"),
    G_EVENTS: (G_EVENTS, "GeneratedEvents"),
    "GeneratedEvents": ("GeneratedEvents", G_EVENTS),
    S_EVENTS: (S_EVENTS, "SelectedEvents", "Events"),
    "SelectedEvents": ("SelectedEvents", S_EVENTS, "Events"),
    S_PARTICLES: (S_PARTICLES, "SelectedParticles"),
    "SelectedParticles": ("SelectedParticles", S_PARTICLES),
}


def candidates(requested: str) -> tuple[str, ...]:
    """Return the requested name followed by schema-compatible aliases."""
    return _ALIASES.get(requested, (requested,))


def resolve(root_file, requested: str) -> str:
    """Return the first available compatible tree, or the requested name."""
    for name in candidates(requested):
        if root_file.Get(name):
            return name
    return requested
