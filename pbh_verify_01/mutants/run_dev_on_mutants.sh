#!/bin/bash
# Run the CURRENT developer test suite (tests/dev) against every mutant of a code version (+ its BASE copy).
# usage: mutants/run_dev_on_mutants.sh <mutant_root> <out_file>
cd "$(dirname "$0")/.."
MR=${1:-mutants/v0.1}; out=${2:-results/mutants_dev.txt}; : > $out
for d in $MR/M*/ $MR/BASE/; do
  m=$(basename $d)
  res=$(PYTHONPATH=$d/src timeout 600 python3 -m pytest tests/dev -q -p no:cacheprovider 2>&1 | tail -1)
  fails=$(PYTHONPATH=$d/src timeout 600 python3 -m pytest tests/dev -q -p no:cacheprovider 2>&1 | grep ^FAILED | sed 's/ - .*//' | tr '\n' ' ')
  echo "$m | $res | $fails" | tee -a $out
done
