from functools import wraps
def handle_interpolation_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if "out of bounds" in str(e):
                raise AssertionError("Interpolation error: out of bounds")
            raise
    return wrapper