#!/usr/bin/env python3
"""Verify the release; optionally rerun frozen-output analysis and PDF compilation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ANALYSES = [
    'synthetic_benchmark_v2/analyze_ccfa_revision.py',
    'synthetic_benchmark_v2/analyze_ccfa_supplements.py',
    'synthetic_benchmark_v2/analyze_ccfa_labels_external.py',
    'synthetic_benchmark_v2/analyze_ccfa_secondary.py',
    'synthetic_benchmark_v2/analyze_ccfa_legacy_comparison.py',
    'synthetic_benchmark_v2/analyze_referee_revision.py',
    'synthetic_benchmark_v2/analyze_constraint_transfer.py',
    'synthetic_benchmark_v2/verify_ccfa_completion.py',
    'synthetic_benchmark_v2/analyze_acceptance_revision.py',
    'synthetic_benchmark_v2/write_acceptance_tables.py',
    'synthetic_benchmark_v2/verify_acceptance_revision.py',
    'figures/make_manuscript_figures.py',
    'figures/make_manuscript_result_figures.py',
    'figures/make_acceptance_figures.py',
    'synthetic_benchmark_v2/finalize_acceptance_manifest.py',
    'synthetic_benchmark_v2/analyze_unified_revision.py',
    'synthetic_benchmark_v2/audit_unified_corrections.py',
    'synthetic_benchmark_v2/verify_unified_revision.py',
    'figures/make_unified_figures.py',
    'figures/make_presentation_restoration.py',
    'synthetic_benchmark_v2/finalize_unified_manifest.py',
]


def verify():
    manifest = json.loads((ROOT / 'SHA256SUMS.json').read_text())
    failures = []
    for name, expected in manifest.items():
        path = ROOT / name
        if not path.is_file():
            failures.append({'file': name, 'status': 'missing'})
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            failures.append({'file': name, 'status': 'hash_mismatch'})
    print(json.dumps({'files': len(manifest), 'all_match': not failures,
                      'failures': failures}, indent=2), flush=True)
    if failures:
        raise SystemExit('Use a fresh release checkout before reproduction.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--analysis', action='store_true', help='Recompute using included predictions; no inference.')
    parser.add_argument('--paper', action='store_true', help='Build PDF under reproduced/paper/.')
    args = parser.parse_args()
    if args.verify_only and (args.analysis or args.paper):
        parser.error('--verify-only cannot be combined with execution flags')
    verify()
    if args.analysis:
        for name in ANALYSES:
            print('Running ' + name, flush=True)
            subprocess.run([sys.executable, name], cwd=ROOT, check=True)
    if args.paper:
        out = ROOT / 'reproduced/paper'
        out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, 'BIBINPUTS': str(ROOT) + ':' + os.environ.get('BIBINPUTS', '')}
        latex = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                 '-output-directory', str(out), 'IEEE-conference-template-062824.tex']
        for command, cwd in [(latex, ROOT), (['bibtex', 'IEEE-conference-template-062824'], out),
                             (latex, ROOT), (latex, ROOT)]:
            subprocess.run(command, cwd=cwd, env=env, check=True)
        from manuscript_page_accounting import page_accounting
        print(json.dumps(page_accounting(out / 'IEEE-conference-template-062824.pdf'), indent=2))


if __name__ == '__main__':
    main()
