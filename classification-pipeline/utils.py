from typing import List, Tuple


def generate_batch_data(x, batch_size: int) -> Tuple[List, int]:
    """
    Generates batches of given size.
    :param x: a list of entries.
    :param batch_size: the size of a single batch
    :return: a list for the current batch and the batch id
    """
    i, batch = 0, 0
    for batch, i in enumerate(range(0, len(x) - batch_size, batch_size), 1):
        x_batch = x[i: i + batch_size]
        yield x_batch, batch
    if i + batch_size < len(x):
        yield x[i + batch_size:], batch + 1
    if batch == 0:
        yield x, 1
