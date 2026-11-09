#!/usr/bin/env python3
"""Compare a local checkpoint with the files fingerprinted in the original run."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    entries = json.loads((ROOT / 'MODEL_SOURCES.json').read_text())['models']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=[entry['annotator_id'] for entry in entries])
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    entry = next(entry for entry in entries if entry['annotator_id'] == args.model)
    recorded = entry['recorded_files']
    files = [recorded['config'], *recorded['weights'], *recorded['tokenizer_files']]
    failures = []
    for record in files:
        path = args.checkpoint / record['name']
        if not path.is_file():
            failures.append({'file': record['name'], 'status': 'missing'})
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != record['sha256']:
            failures.append({'file': record['name'], 'status': 'hash_mismatch'})
    print(json.dumps({'model': args.model, 'recorded_files': len(files),
                      'all_match': not failures, 'failures': failures}, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
