from __future__ import annotations

import os
import sys
from typing import Any


def tvm_import() -> Any:
    """Import TVM.

    Preference order:
    1) Normal `import tvm` (installed package)
    2) Add `$TVM_HOME/python` (default: /root/myfold/tvm/python) to sys.path

    Returns the imported `tvm` module.
    """

    try:
        import tvm  # type: ignore

        return tvm
    except ModuleNotFoundError:
        tvm_home = os.environ.get("TVM_HOME", "/root/myfold/tvm")
        tvm_python = os.path.join(tvm_home, "python")
        if os.path.isdir(tvm_python) and tvm_python not in sys.path:
            sys.path.insert(0, tvm_python)
        import tvm  # type: ignore

        return tvm
