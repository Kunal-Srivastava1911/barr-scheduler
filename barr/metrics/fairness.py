def jain_index(values):
    """Jain's fairness index J = (sum x)^2 / (n * sum x^2); 1 when all are equal."""
    values = list(values)
    if not values:
        return 1.0
    s = sum(values)
    s2 = sum(v * v for v in values)
    return (s * s) / (len(values) * s2) if s2 > 0 else 1.0
