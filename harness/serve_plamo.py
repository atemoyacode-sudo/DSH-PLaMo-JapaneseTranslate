"""Serve PLaMo 3 Translate (MLX-4bit) with the model's own serve.py, under memory limits.

  python -m harness.serve_plamo --model <plamo-3-translate-0.8b-MLX-4bit dir> --port 8091

The model's bundled serve.py (an mlx-lm server with PFN's fixes) does the serving; this wrapper only caps MLX's
memory first (mlx-lm otherwise wires memory up to the recommended working set, which leaves little room for the
main model next to it). Limits can be changed with PLAMO_WIRED_GB, PLAMO_CACHE_GB and PLAMO_MEM_GB.
Defaults suit the 0.8B model (about 1 GB of weights). Loads the model's bundled code (--trust-remote-code), as
its card says.
"""

from __future__ import annotations

import os
import sys

GB = 1 << 30


def main() -> None:
    args = sys.argv[1:]
    if "--model" not in args:
        sys.exit(__doc__)
    model_dir = os.path.abspath(args[args.index("--model") + 1])
    import mlx.core as mx
    import mlx_lm.server

    wired, cache, mem = (float(os.environ.get(k, d)) for k, d in
                         (("PLAMO_WIRED_GB", 2), ("PLAMO_CACHE_GB", 0.5), ("PLAMO_MEM_GB", 4)))
    mlx_lm.server.maybe_set_recommended_wired_limit = lambda: mx.set_wired_limit(int(wired * GB))
    mx.set_cache_limit(int(cache * GB))
    mx.set_memory_limit(int(mem * GB))
    print(f"memory limits: wired {wired:g} GB, buffer cache {cache:g} GB, memory {mem:g} GB", flush=True)
    sys.path.insert(0, model_dir)  # the model's serve.py imports its server.py and serving_memory.py
    import serve  # noqa: E402  (PFN's serve.py from the model directory)

    sys.argv = [os.path.join(model_dir, "serve.py")] + args
    if "--trust-remote-code" not in sys.argv:
        sys.argv.append("--trust-remote-code")
    for flag, value in (("--host", "127.0.0.1"), ("--port", "8091"), ("--decode-concurrency", "2")):
        if flag not in sys.argv:
            sys.argv += [flag, value]
    serve.main()


if __name__ == "__main__":
    main()
