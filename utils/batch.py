# utils/batch.py
from itertools import islice
from typing import Iterable, List, Iterator, TypeVar

T = TypeVar("T")

def batched(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    """Yield successive n-sized batches from iterable."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

