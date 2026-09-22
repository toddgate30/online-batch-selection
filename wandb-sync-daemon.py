import subprocess
import time
import argparse
import glob
import os

EXPERIMENTS_ROOT = './experiments'

def sync(subdirs):
    start = time.time()

    if subdirs:
        roots = []
        for subdir in subdirs:
            roots.extend(glob.glob(os.path.join(EXPERIMENTS_ROOT, subdir)))
    else:
        roots = [EXPERIMENTS_ROOT]

    run_dir_to_offline_dirs = {}
    for root in roots:
        for offline_run_dir in glob.glob(
            os.path.join(root, '**', 'wandb', 'offline-run-*'), recursive=True
        ):
            run_dir = os.path.dirname(os.path.dirname(offline_run_dir))
            if "slurm_history" in run_dir.split(os.sep):
                continue
            run_dir_to_offline_dirs.setdefault(run_dir, []).append(offline_run_dir)

    to_sync = []
    finished_in_call = []
    n_skipped = 0
    for run_dir, offline_run_dirs in run_dir_to_offline_dirs.items():
        finished = os.path.isfile(os.path.join(run_dir, 'FINISHED'))
        synced = os.path.isfile(os.path.join(run_dir, 'SYNCED'))

        if finished and synced:
            n_skipped += 1
            continue

        to_sync.extend(offline_run_dirs)
        if finished:
            finished_in_call.append(run_dir)

    if to_sync:
        cmd = ["wandb", "sync", *to_sync]
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as proc:
            for line in proc.stdout:
                print(line, end="")
        returncode = proc.returncode

        if returncode == 0:
            for run_dir in finished_in_call:
                with open(os.path.join(run_dir, 'SYNCED'), 'w') as f:
                    pass
    else:
        abs_roots = [os.path.abspath(r) for r in roots]
        print(f"No data found to sync (searched in {abs_roots if len(abs_roots) > 0 else subdirs})")

    print(f"Skipped {n_skipped} finished+synced run dir(s)")

    return time.time() - start

def sync_daemon(subdirs, interval_sec=30):
    while True:
        print(f"Syncing... ({interval_sec}s interval)")
        elapsed = sync(subdirs)
        print(f"Sync took {elapsed:.2f}s")

        sleep_time = max(0, interval_sec - elapsed)
        end_time = time.time() + sleep_time

        # Display time remaining before next sync
        while True:
            remaining = end_time - time.time()
            if remaining <= 0:
                print(f"\rNext sync in 0.0s          ", flush=True)
                break

            print(f"\rNext sync in {remaining:.1f}s          ", end="", flush=True)
            time.sleep(0.1)

        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'subdirs', type=str, nargs='*',
        help=f"Subdirectory name(s)/glob pattern(s) under {EXPERIMENTS_ROOT}/. "
             "Each is recursively searched for run dirs to sync. "
             "If omitted, recursively syncs all run dirs under it.",
    )
    parser.add_argument('--interval', type=int, default=15)
    args = parser.parse_args()
    sync_daemon(args.subdirs, args.interval)
