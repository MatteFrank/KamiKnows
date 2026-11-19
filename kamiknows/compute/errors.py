"""The single error type of the compute package."""


class NotComputable(Exception):
    """The requested quantity cannot be derived from the available values.

    Raised instead of returning an approximation, so the caller can tell the
    user why the answer is missing: plot not digitized, point outside the
    domain, incompatible units, and so on.
    """
