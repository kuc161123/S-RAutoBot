"""Start the Apex cloud runtime; retain an explicit legacy rollback entrypoint."""

import os
import runpy
from pathlib import Path


def main():
    runtime = os.environ.get("AUTOBOT_RUNTIME", "apex")
    if runtime == "legacy":
        runpy.run_path(
            str(Path(__file__).with_name("legacy_main.py")), run_name="__main__"
        )
    elif runtime == "apex":
        from apex_bot.__main__ import main as start_apex

        start_apex()
    else:
        raise ValueError("AUTOBOT_RUNTIME must be apex or legacy")


if __name__ == "__main__":
    main()
