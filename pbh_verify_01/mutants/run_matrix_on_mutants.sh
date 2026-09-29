#!/bin/bash
# Run a FROZEN copy of the independent matrix (git tag $2, extracted to $3) against every mutant.
# usage: mutants/run_matrix_on_mutants.sh <label> <frozen_dir> <mutant_root (e.g. mutants/v0.3)>
cd "$(dirname "$0")/.."
P=$(pwd); tag=$1; F=$2; MR=${3:-mutants/v0.1}
mkdir -p results/mutants_matrix_${tag}
for d in $MR/M*/ $MR/BASE/; do
  m=$(basename $d)
  SPHDIAG_PATH=$P/$d/src timeout 900 python3 $F/tests/independent/run_matrix.py --out $P/results/mutants_matrix_${tag}/$m.json > /dev/null 2>&1
  python3 -c "
import json;d=json.load(open('results/mutants_matrix_${tag}/$m.json'))
f=[c['id'] for c in d['results'] if c['status']!='PASS'];print('$m',d['summary'],' '.join(f))"
done
