#!/bin/sh
cd "$(dirname "$0")/scripts" || exit 1
export PYTHONIOENCODING=utf-8
set -x
python 03_load_db.py --force            || exit 1
python 04_pick_queries.py               || exit 1
python 05_run_experiments.py --sizes small  --repeats 5 || exit 1
python 05_run_experiments.py --sizes medium --repeats 5 || exit 1
python 05_run_experiments.py --sizes large  --repeats 5 --build-repeats-large 2 \
    --indexes fts_gin,fts_gist,fts_rum,vec_hnsw,vec_ivfflat,fts_gin_nofu,none || exit 1
python 05_run_experiments.py --sizes large  --repeats 5 --build-repeats-large 1 \
    --indexes trgm_gin,trgm_gist || exit 1
