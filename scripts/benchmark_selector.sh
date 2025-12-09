#!/bin/bash
# UVG Selector Benchmark Script
# Usage: bash scripts/benchmark_selector.sh /path/to/clips "prompt text"

set -e

# Arguments
CLIPS_DIR="${1:-./test_clips}"
PROMPT="${2:-a beautiful nature scene}"

echo "========================================"
echo "UVG Selector Benchmark"
echo "========================================"
echo "Clips Directory: $CLIPS_DIR"
echo "Prompt: $PROMPT"
echo ""

# Find up to 10 mp4 files
CLIP_FILES=$(find "$CLIPS_DIR" -maxdepth 1 -name "*.mp4" | head -10)
CLIP_COUNT=$(echo "$CLIP_FILES" | wc -l)

if [ -z "$CLIP_FILES" ]; then
    echo "ERROR: No .mp4 files found in $CLIPS_DIR"
    exit 1
fi

echo "Found $CLIP_COUNT clip(s)"
echo ""

# Create Python benchmark script inline
python3 << EOF
import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from uvg_selector.clip_selector import rank_clips
    from uvg_selector.cache import get_cache_stats, clear_cache
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure uvg_selector is in the Python path")
    sys.exit(1)

# Get clip files
clips_dir = Path("$CLIPS_DIR")
clip_files = sorted(clips_dir.glob("*.mp4"))[:10]
clip_paths = [str(p) for p in clip_files]

prompt = "$PROMPT"

print(f"Processing {len(clip_paths)} clips...")
print("")

# Clear cache for fresh benchmark
clear_cache()

# Run ranking
start_time = time.time()

try:
    results = rank_clips(prompt, clip_paths, top_k=5, use_cache=False)
    elapsed = time.time() - start_time
    success = True
except Exception as e:
    results = []
    elapsed = time.time() - start_time
    success = False
    error_msg = str(e)

# Build report
report = {
    "prompt": prompt,
    "clips_processed": len(clip_paths),
    "elapsed_seconds": round(elapsed, 3),
    "success": success,
    "top_picks": []
}

if success:
    for i, r in enumerate(results[:5]):
        report["top_picks"].append({
            "rank": i + 1,
            "path": r["path"],
            "score": round(r.get("final_score", 0), 4),
            "signals": {k: round(v, 4) if isinstance(v, float) else v 
                       for k, v in r.get("signals", {}).items()}
        })
else:
    report["error"] = error_msg

# Cache stats
cache_stats = get_cache_stats()
report["cache"] = cache_stats

# Print report
print("========================================")
print("RESULTS")
print("========================================")
print(f"Time: {elapsed:.2f}s ({len(clip_paths)/max(0.001,elapsed):.1f} clips/sec)")
print("")

if success:
    print("Top Picks:")
    for pick in report["top_picks"]:
        print(f"  {pick['rank']}. {Path(pick['path']).name}: {pick['score']:.4f}")
else:
    print(f"ERROR: {report.get('error', 'Unknown error')}")

print("")

# Save JSON report
report_path = "selector_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"Report saved to: {report_path}")
EOF

echo ""
echo "========================================"
echo "Benchmark complete"
echo "========================================"
