#!/usr/bin/env bash
# Refresh this public mirror from the working repositories in /root/PBH on the VPS.
# Only the scientific libraries are published (path lists below); their history is kept, restricted to
# those paths, and masked.  Each source repo is fresh-cloned and rewritten with git filter-repo; the
# rewrite is deterministic, so old commits keep their hashes and a sync only merges new commits.
# Masking rules and the author mailmap live outside the repo (they contain what they hide):
#   $PBH_MASK_DIR/mask_rules.txt   old==>new, one per line (filter-repo --replace-text format)
#   $PBH_MASK_DIR/mailmap
# tools/path_rewrites.txt (public) rewrites references to files that are not published.
# README*, LICENSE, LICENSING.md, CITATION.cff, tools/ and pbhgr/docs/areal_cmc_gauge.md are maintained here,
# not synced.
# Usage: tools/sync_from_workspace.sh [--push]
set -euo pipefail
SRC=${PBH_WORKSPACE:-/root/PBH}
PRIV=${PBH_MASK_DIR:-/root/pbh-github-private}
RULES=$PRIV/mask_rules.txt
WORK=$PRIV/work
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
[ -f "$RULES" ] && [ -f "$PRIV/mailmap" ] || { echo "missing $RULES or $PRIV/mailmap" >&2; exit 1; }

# workspace repo -> published subdirectory, and the paths published from it
PBHGR_PATHS=(
  .gitignore Makefile pyproject.toml LICENSE LICENSES/
  pbhgr/ csrc/ tests/                                   # Python package, C kernels, pytest suite
  grteclyn/PBHVlasov/ grchombo/PBHCosmo/                # 3D GR modules
  scripts/grchombo/                                     # their build script, initial data, analysis
  scripts/v1_ladder.py scripts/v1_static_test.py scripts/pm3d_peak.py scripts/ellipsoidal_collapse.py
  docs/stage2_vlasov_design.md docs/stage1b_design.md docs/grchombo_ax102.md
)
SPHDIAG_PATHS=(
  src/sphdiag/ tests/dev/ physics_contract.md derivation.md derivation/
)

publish() {   # publish <workspace repo> <subdir> <extra filter-repo args...> -- <paths...>
  local name=$1 sub=$2; shift 2
  local extra=(); while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
  local sel=(); for p in "$@"; do sel+=(--path "$p"); done
  rm -rf "${WORK:?}/$name"
  git clone -q --no-local --no-tags "$SRC/$name" "$WORK/$name"
  git -C "$WORK/$name" filter-repo --force --quiet "${sel[@]}"
  if [ ${#extra[@]} -gt 0 ]; then git -C "$WORK/$name" filter-repo --force --quiet "${extra[@]}"; fi
  git -C "$WORK/$name" filter-repo --force --quiet \
      --replace-text "$WORK/rules.txt" --replace-message "$WORK/rules.txt" --mailmap "$PRIV/mailmap" \
      --to-subdirectory-filter "$sub"
  git fetch -q --no-tags "$WORK/$name" "+master:refs/heads/import/$sub"
  git merge -q --allow-unrelated-histories --no-edit -m "Merge $sub from workspace" "import/$sub"
}

mkdir -p "$WORK"
cat "$RULES" tools/path_rewrites.txt > "$WORK/rules.txt"
publish pbhgr pbhgr -- "${PBHGR_PATHS[@]}"
publish pbh_verify_01 sphdiag --path-rename src/sphdiag/README.md:README.md -- "${SPHDIAG_PATHS[@]}"

# final leak check on the whole published history
leaks=$(git log --all -p --format='%an <%ae>%n%B' | grep -c -F -f <(cut -d'>' -f1 "$RULES" | sed 's/==$//') || true)
if [ "$leaks" != 0 ]; then echo "LEAK CHECK FAILED: $leaks masked strings still present" >&2; exit 2; fi
git log --oneline -3
if [ "${1:-}" = "--push" ]; then
  git push origin main
fi
