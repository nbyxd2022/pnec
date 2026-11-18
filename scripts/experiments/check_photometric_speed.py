"""Compile and compare the optimized SSD search with the frozen scalar implementation."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


REFERENCE_COMMIT = "1b5ee611ee341fb9307fe35b4726c9cd5d87c52c"
HEADER = "include/rel_pose_estimation/photometric_increment.h"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    old_header = subprocess.check_output(
        ["git", "show", f"{REFERENCE_COMMIT}:{HEADER}"], cwd=repo)
    namespace = b"namespace pnec::rel_pose_estimation::photometric {"
    if old_header.count(namespace) != 1:
        raise ValueError("Unexpected reference header namespace")
    old_header = old_header.replace(namespace, namespace.replace(
        b"photometric {", b"photometric_reference {"))
    guard = b"PNEC_PHOTOMETRIC_INCREMENT_H_"
    old_header = old_header.replace(guard, b"PNEC_PHOTOMETRIC_INCREMENT_REFERENCE_H_")
    with TemporaryDirectory(prefix="pnec-photometric-check-") as temporary:
        work = Path(temporary)
        (work / "photometric_increment_reference.h").write_bytes(old_header)
        executable = work / "check_photometric_speed"
        command = [args.cxx, "-std=c++17", "-O3", "-march=native",
                   "-I", str(repo / "include/rel_pose_estimation"), "-I", str(work),
                   str(repo / "scripts/experiments/check_photometric_speed.cc"),
                   "-o", str(executable)]
        subprocess.run(command, cwd=repo, check=True)
        result = json.loads(subprocess.check_output([str(executable)], text=True))
    result["reference_commit"] = REFERENCE_COMMIT
    result["current_header_sha256"] = sha256((repo / HEADER).read_bytes()).hexdigest()
    result["compiler"] = subprocess.check_output(
        [args.cxx, "--version"], text=True).splitlines()[0]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
