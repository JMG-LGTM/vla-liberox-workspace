#!/bin/bash
# Run an IDQL evaluation (all halves in the config) through the platform's unchanged evaluation loop.
# Usage: run_eval.sh CONFIG.json      (run prepare_eval.py first)
# Holds the platform GPU lock for the whole run, so no platform GPU job starts meanwhile.
# Results stay in the config's results_dir; nothing is written to the platform's evaluations/ or jobs/.
set -uo pipefail
CONFIG=$(realpath "${1:?usage: run_eval.sh CONFIG.json}")
HERE=$(cd "$(dirname "$0")" && pwd)
get() { python3 -c "import json,sys; v=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$CONFIG" "$1"; }
SNAP=$(get "v.get('platform_env_snapshot') or ''")
if [ -n "$SNAP" ]; then  # same environment as platform jobs: a copy of /proc/<platform pid>/environ
  while IFS= read -r -d '' kv; do export "$kv"; done < "$SNAP"
fi
export IDQL_EVAL_CONFIG=$CONFIG
WS=$(get "v['workspace']"); RES=$(get "v['results_dir']"); ENV=$(get "v.get('platform_conda_env','vla-liberox')")
exec 9>>"$WS/dataset-root/projects/libero_x_vla/.gpu-task.lock"
flock -n 9 || { echo "GPU lock busy"; exit 2; }
cd "$WS/liberox-vla-adapter-terminal"
for half in $(get "' '.join(h['name'] for h in v['halves'])"); do
  [ -f "$RES/$half/evaluation.json" ] && grep -q '"COMPLETED"' "$RES/$half/evaluation.json" && continue
  echo "$(date -Is) start $half"
  conda run --no-capture-output -n "$ENV" python -B "$HERE/run_idql_evaluation.py" \
    --config "$RES/$half/effective_config.yaml" --runtime-contract "$RES/runtime_contract.json" > "$RES/$half/run.log" 2>&1
  rc=$?; echo "$(date -Is) end $half rc=$rc"; [ $rc -eq 0 ] || exit $rc
done
