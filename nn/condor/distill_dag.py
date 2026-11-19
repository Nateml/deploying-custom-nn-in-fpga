#!/usr/bin/env python3
"""Write a DAGMan file that trains one distill.py student per seed as a chain of
short (4 h) condor jobs.

Each node runs distill.py with --time-limit, which saves its state and exits with
code 3 before the 4 h limit; DAGMan then resubmits the node, which resumes. A crash
in Python (exit code 1) is not retried. Runs on the host (plain python3), from nn/:

    python3 condor/distill_dag.py kd --seeds 1 2 3 4 5 -- --lr 1e-4
    python3 condor/distill_dag.py hard --seeds 1 2 3 4 5 -- --lr 1e-4 --alpha 0
    condor_submit_dag /data/gravwav/nmacdona/earlywarningfpgas/output/condor/dags/kd.dag

Everything after -- is passed to distill.py.
"""

import argparse
import os
import sys

NN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAG_DIR = "/data/gravwav/nmacdona/earlywarningfpgas/output/condor/dags/"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("name", help="DAG name, also the prefix of the node names")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--time-limit", type=float, default=3.5, help="hours per job (short: 4 h)"
    )
    parser.add_argument(
        "--retries", type=int, default=6, help="resubmissions per seed after the first"
    )
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    distill_args = argv[split + 1 :]
    if "--seeds" in distill_args or "--time-limit" in distill_args:
        parser.error("--seeds and --time-limit are set per node, not after --")

    os.makedirs(DAG_DIR, exist_ok=True)
    dag = os.path.join(DAG_DIR, f"{args.name}.dag")
    if os.path.exists(dag):
        parser.error(
            f"{dag} exists (DAGMan keeps its logs next to it), pick a new name"
        )
    with open(dag, "w") as f:
        for seed in args.seeds:
            node = f"{args.name}_seed{seed}"
            command = " ".join(
                ["distill.py", *distill_args, "--seeds", str(seed)]
                + ["--time-limit", str(args.time_limit)]
            )
            f.write(f"JOB {node} {NN_DIR}/condor/job.sub DIR {NN_DIR}\n")
            f.write(f'VARS {node} args="{command}" category="short"\n')
            f.write(f"RETRY {node} {args.retries} UNLESS-EXIT 1\n")
    print(f"Wrote {dag}")
    print(f"Submit with: condor_submit_dag {dag}")


if __name__ == "__main__":
    main()
