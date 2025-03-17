from tqdm import tqdm

def progress_bar(total_iterations, description):
    # The problem is that you're passing a range object to tqdm
    # but tqdm expects an integer for its 'total' parameter

    # Check if total_iterations is a range object
    if isinstance(total_iterations, range):
        # Convert range to its length
        total_count = len(total_iterations)
        iteration_source = total_iterations
    else:
        # Use as-is if it's already an integer
        total_count = total_iterations
        iteration_source = range(total_iterations)

    def update_color(pbar):
        if pbar.n == total_count:
            pbar.colour = 'green'
        else:
            pbar.colour = 'red'

    read_bar_format = "%s{l_bar}%s{bar}%s{r_bar}%s" % (
        "\033[37m", "\033[37m", "\033[37m", "\033[0m"
    )

    with tqdm(total=total_count, desc=description, bar_format=read_bar_format, unit='iteration', ncols=80, colour='white') as pbar:
        for i in iteration_source:
            yield i
            pbar.update(1)
            update_color(pbar)

    update_color(pbar)