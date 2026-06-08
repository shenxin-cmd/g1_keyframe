#!/usr/bin/env bash
# 查看批量采集进度
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BATCH_ROOT="${G1_BATCH_ROOT:-$REPO/batch_data}"
MANIFEST="$BATCH_ROOT/logs/manifest.jsonl"
EVENTS="$BATCH_ROOT/logs/events.jsonl"

echo "=== G1 批量采集状态 ==="
echo "数据目录: $BATCH_ROOT"
echo ""

if [[ -f "$BATCH_ROOT/state/runner.lock" ]]; then
  echo "Runner 锁: $(cat "$BATCH_ROOT/state/runner.lock")"
else
  echo "Runner 锁: (无，未在运行或已结束)"
fi

if [[ -f "$BATCH_ROOT/state/checkpoint.json" ]]; then
  echo "最近 checkpoint:"
  cat "$BATCH_ROOT/state/checkpoint.json"
  echo ""
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest 不存在，任务尚未开始。"
  exit 0
fi

python3 - <<PY
import json
from collections import Counter

manifest_path = "$MANIFEST"
latest = {}
with open(manifest_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        latest[rec["job_id"]] = rec

counts = Counter(r.get("status", "?") for r in latest.values())
print(f"Job 总数(去重): {len(latest)}")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")

pass_clips = sum(int(r.get("n_pass", 0)) for r in latest.values())
reject_clips = sum(int(r.get("n_reject", 0)) for r in latest.values())
print(f"片段累计: pass={pass_clips} reject={reject_clips}")

failed = [r for r in latest.values() if r.get("status") == "failed"]
if failed:
    print(f"\n最近失败 ({min(5, len(failed))} 条):")
    for r in failed[-5:]:
        print(f"  {r['job_id']}: {r.get('error', '')[:120]}")

recent = sorted(latest.values(), key=lambda x: x.get("updated_at", x.get("finished_at", "")), reverse=True)[:5]
print("\n最近更新:")
for r in recent:
    print(
        f"  {r.get('job_id')} | {r.get('status')} | "
        f"pass={r.get('n_pass', '-')} reject={r.get('n_reject', '-')}"
    )
PY

echo ""
echo "clips_obs 文件数: $(find "$BATCH_ROOT/clips_obs" -name '*_obs.npz' 2>/dev/null | wc -l)"
echo "rejected 记录数: $(find "$BATCH_ROOT/rejected" -name '*.reject.json' 2>/dev/null | wc -l)"
echo ""
echo "实时日志: tail -f $BATCH_ROOT/logs/run.log"
if [[ -f "$EVENTS" ]]; then
  echo "最近事件:"
  tail -n 5 "$EVENTS"
fi
