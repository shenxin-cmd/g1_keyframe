#!/usr/bin/env bash
# 在 tmux 中启动 G1 批量轨迹采集（可 resume）
set -euo pipefail

SESSION="${G1_BATCH_SESSION:-g1_batch}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BATCH_ROOT="${G1_BATCH_ROOT:-$REPO/batch_data}"

mkdir -p "$BATCH_ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux 会话 '$SESSION' 已存在。附加: tmux attach -t $SESSION"
  echo "或先结束: tmux kill-session -t $SESSION"
  exit 1
fi

# 并行 worker 数：默认 CPU 核数-1（至少 1）。各 worker 可同时跑不同形状/seed，不牺牲 IK 精度。
# 例: G1_BATCH_WORKERS=8 bash scripts/run_batch_tmux.sh
WORKERS="${G1_BATCH_WORKERS:-$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) - 1 ))}"
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
# 形状平面：yz(默认) | xy | xz。例: G1_SHAPE_PLANE=xy bash scripts/run_batch_tmux.sh
SHAPE_PLANE="${G1_SHAPE_PLANE:-yz}"
# 可选快速 IK（略降精度）：G1_FAST_IK=1 bash scripts/run_batch_tmux.sh
FAST_IK_FLAG=""
if [ "${G1_FAST_IK:-0}" = "1" ]; then FAST_IK_FLAG="--fast-ik"; fi

CMD="cd '$REPO' && \
  source \"\$(conda info --base)/etc/profile.d/conda.sh\" && \
  conda activate key && \
  export G1_BATCH_ROOT='$BATCH_ROOT' && \
  python g1_batch_runner.py --resume --force --no-filter $FAST_IK_FLAG --workers $WORKERS --shape-plane '$SHAPE_PLANE' --layers 1 --frames-per-ring 300 --fps 30 --seed-count 100 --batch-root '$BATCH_ROOT' 2>&1 | tee -a '$BATCH_ROOT/logs/run.log'; \
  echo ''; echo '[g1_batch] 批量任务已结束。窗口保持打开，exit 可关闭。'; \
  exec bash"

tmux new-session -d -s "$SESSION" bash -lc "$CMD"
echo "已启动 tmux 会话: $SESSION (plane=$SHAPE_PLANE, workers=$WORKERS${G1_FAST_IK:+", fast-ik"})"
echo "查看: tmux attach -t $SESSION"
echo "日志: tail -f $BATCH_ROOT/logs/run.log"
echo "调并行度: G1_BATCH_WORKERS=4 bash scripts/run_batch_tmux.sh"
echo "换平面: G1_SHAPE_PLANE=xy bash scripts/run_batch_tmux.sh"
echo "要快速 IK: G1_FAST_IK=1 bash scripts/run_batch_tmux.sh"
