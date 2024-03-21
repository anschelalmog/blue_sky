def handle_interpolation_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            print(f"Interpolation error occurred: {str(e)}")
            return None
    return wrapper