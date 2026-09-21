#!/usr/bin/env python3
"""Freeze an explicit release policy contract before collecting evidence.

This does not build a catalog or enable a public channel. Existing files are
never overwritten. Calibrated runtime/builder/UI integration remains required.
"""
import argparse
import json
from pathlib import Path
from streaming_release_policy import STRICT, CALIBRATED, freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True, choices=(STRICT, CALIBRATED))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(freeze(args.policy), stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
