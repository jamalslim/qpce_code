#!/usr/bin/env python3
"""
Reproduce every simulation number in the manuscript, then check them.

Run this and nothing else if what you want is to confirm the paper. It trains
nothing. It loads the shipped checkpoint, recomputes each quantity, writes
outputs/numbers.tex, and then compares that file against the values printed in
the manuscript. The comparison is the point. A script that merely runs proves
very little, so this one ends by telling you how many numbers agreed and names
any that did not.

    python scripts/reproduce.py                  # check against the shipped reference
    python scripts/reproduce.py --tex paper.tex  # check against your own manuscript

The device measurements reported in the manuscript come from the archived
hardware run and are reported here as skipped rather than silently ignored.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(script, *args):
    """Run one of the analysis scripts and fail loudly if it does not finish."""
    cmd = [sys.executable, os.path.join(HERE, script), *args]
    print(f"  running {script} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"{script} failed")


def macros_from(path, command):
    """Pull the macro table out of a LaTeX file."""
    pat = r'\\' + command + r'\{\\(num[A-Za-z]+)\}\{([^}]*)\}'
    return dict(re.findall(pat, open(path).read()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=None,
                    help="manuscript to check against, defaults to the shipped reference")
    a = ap.parse_args()

    print("Recomputing from the checkpoint in outputs/checkpoint")
    run("paper_numbers.py")
    run("bell_generative_advantage.py")

    generated = macros_from(os.path.join(ROOT, "outputs", "numbers.tex"), "newcommand")

    # The Bell macros are derived from the certification result rather than
    # written by paper_numbers.py, so assemble them here in the same format.
    bell = json.load(open(os.path.join(ROOT, "outputs", "bell_generative.json")))
    pairs = bell["pairs"]["full"]
    smax = max(v["max"] for v in pairs.values())
    frac = sum(v["frac_violating"] for v in pairs.values()) / len(pairs)
    shot = bell["finite_shot"][0]
    generated.update({
        "numBellSmax":   f"{smax:.2f}",
        "numBellFrac":   f"{frac * 100:.0f}\\%",
        "numBellShot":   f"{shot['S_hat']:.3f}",
        "numBellShotSe": f"{shot['se']:.3f}",
        "numBellSigma":  f"{shot['sigma']:.0f}",
    })

    reference = a.tex or os.path.join(ROOT, "outputs", "manuscript_reference.tex")
    if not os.path.exists(reference):
        print("\nNo manuscript to compare against, wrote outputs/numbers.tex only.")
        return
    expected = macros_from(reference, "providecommand" if a.tex is None else "providecommand")
    if not expected:
        expected = macros_from(reference, "newcommand")

    hardware = {k for k in expected if k.startswith("numHw") or k == "numWidthHw"}
    checked = sorted(k for k in expected if k not in hardware)

    agree, differ, absent = [], [], []
    for k in checked:
        if k not in generated:
            absent.append(k)
        elif generated[k].strip() == expected[k].strip():
            agree.append(k)
        else:
            differ.append((k, expected[k], generated[k]))

    print(f"\n{len(agree)} of {len(checked)} simulation numbers reproduced exactly")
    if differ:
        print(f"\n{len(differ)} disagree:")
        for k, want, got in differ:
            print(f"  {k:24s} manuscript {want:>12s}   recomputed {got:>12s}")
    if absent:
        print(f"\n{len(absent)} not produced by this release: {', '.join(absent)}")
    print(f"{len(hardware)} device measurements skipped, they come from the hardware run")

    raise SystemExit(1 if (differ or absent) else 0)


if __name__ == "__main__":
    main()
