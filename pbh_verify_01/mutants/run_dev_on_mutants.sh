#!/bin/bash
# Run the developer test suite (tests/dev, frozen at tag impl-B-v0.1) against every mutant.
cd "$(dirname "$0")/.."
out=results/mutants_dev.txt; : > $out
for d in mutants/M*/; do
  m=$(basename $d)
  res=$(PYTHONPATH=$d/src timeout 600 python3 -m pytest tests/dev -q -p no:cacheprovider 2>&1 | tail -1)
  fails=$(PYTHONPATH=$d/src timeout 600 python3 -m pytest tests/dev -q -p no:cacheprovider 2>&1 | grep ^FAILED | sed 's/ - .*//' | tr '\n' ' ')
  echo "$m | $res | $fails" | tee -a $out
done
