import pathlib
import joblib

__all__ = [
    "path_cache",
    "memory",
]

#: The location on the filesystem where intermediate results are stored.
#:
#: Not the ``~/.spd2026/cache`` of the SPD 2026 talk this repository was
#: forked from, even though the package is still called ``spd2026``.
#: :mod:`joblib` files a result under the name of the function which computed
#: it, and throws the result away when the code of that function changes, so
#: two copies of this package sharing one cache would each throw away the
#: other's results as soon as either was edited. This cache started as a copy
#: of that one, taken on 2026-09-24, so nothing computed for the talk has to
#: be computed again.
path_cache = pathlib.Path.home() / ".solar-research-2026/cache"

#: A representation of the cache which stores intermediate results.
memory = joblib.Memory(location=path_cache, mmap_mode="r", verbose=0)
