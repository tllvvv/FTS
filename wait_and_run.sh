#!/bin/sh
cd "$(dirname "$0")"
while ! grep -q "Готово:" results/embeddings_log.txt 2>/dev/null; do sleep 30; done
sleep 5
./run_pipeline.sh > results/pipeline_log.txt 2>&1
echo "PIPELINE EXIT=$?"
