"""Explicit failures are preferable to silent physical repairs."""


class StateError(ValueError):
    """A structural or representability invariant was violated."""


class ResolutionError(StateError):
    """The grid cannot resolve the selected extended cells."""


class CapacityError(StateError):
    """The bounded reference backend would allocate too much memory."""


class ChargeError(StateError):
    """Periodic electrostatics was requested for an unsupported net charge."""
