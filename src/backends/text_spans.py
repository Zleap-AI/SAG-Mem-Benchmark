"""Lossless bounded character spans; logical source messages retain their identity."""
from collections.abc import Callable, Iterator


def bounded_spans(text: str, count: Callable[[str], int], budget: int,
                  *, prefix: str = "", max_chars: int | None = None) -> Iterator[tuple[int, int]]:
    if budget < 1 or count(prefix) >= budget:
        raise ValueError("Text span budget leaves no room for content")
    start = 0
    window = max_chars or max(256, budget * 4)
    while start < len(text):
        end = min(len(text), start + window)
        if count(prefix + text[start:end]) > budget:
            low, high = start, end
            while low + 1 < high:
                mid = (low + high) // 2
                if count(prefix + text[start:mid]) <= budget:
                    low = mid
                else:
                    high = mid
            end = low
        if end <= start or count(prefix + text[start:end]) > budget:
            raise ValueError("A source character cannot fit in the text span budget")
        yield start, end
        start = end
