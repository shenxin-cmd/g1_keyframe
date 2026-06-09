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

CMD="cd '$REPO' && \
  source \"\$(conda info --base)/etc/profile.d/conda.sh\" && \
  conda activate key && \
  export G1_BATCH_ROOT='$BATCH_ROOT' && \
  python g1_batch_runner.py --resume --force --no-filter --layers 1 --frames-per-ring 300 --seed-count 100 --batch-root '$BATCH_ROOT' 2>&1 | tee -a '$BATCH_ROOT/logs/run.log'; \
  echo ''; echo '[g1_batch] 批量任务已结束。窗口保持打开，exit 可关闭。'; \
  exec bash"

tmux new-session -d -s "$SESSION" bash -lc "$CMD"
echo "已启动 tmux 会话: $SESSION"
echo "查看: tmux attach -t $SESSION"
echo "日志: tail -f $BATCH_ROOT/logs/run.log"
