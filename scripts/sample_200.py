"""
Sample 200 instances from data/instances.jsonl (50 easy, 100 medium, 50 hard)
and write to data/instances_200.jsonl. Run from the CIPHER root directory.
"""
import json
import random
import os
from collections import defaultdict

SEED = 42
COUNTS = {"easy": 50, "medium": 100, "hard": 50}
INPUT  = os.path.join(os.path.dirname(__file__), "..", "data", "instances.jsonl")
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "instances_200.jsonl")

with open(INPUT) as f:
    records = [json.loads(line) for line in f if line.strip()]

by_difficulty = defaultdict(list)
for r in records:
    by_difficulty[r["difficulty"]].append(r)

rng = random.Random(SEED)
sampled = []
for diff, n in COUNTS.items():
    pool = by_difficulty[diff]
    if len(pool) < n:
        raise ValueError(f"Only {len(pool)} {diff} instances, need {n}")
    sampled.extend(rng.sample(pool, n))

rng.shuffle(sampled)

with open(OUTPUT, "w") as f:
    for r in sampled:
        f.write(json.dumps(r) + "\n")

from collections import Counter
breakdown = Counter(r["difficulty"] for r in sampled)
print(f"Wrote {len(sampled)} instances to {OUTPUT}")
print(f"  easy={breakdown['easy']}  medium={breakdown['medium']}  hard={breakdown['hard']}")
