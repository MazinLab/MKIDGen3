


def validate(error=True, **kwargs):
    """

    Args:
        error: Whether to raise an error if validation fails
        **kwargs: arguments for validation

    Returns: dict of valid arguments, if kwargs contains a single key then the value is returned

    """
    return list(kwargs.values())[0] if len(kwargs)==1 else kwargs
