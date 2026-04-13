#!/usr/bin/env python3
"""
Pipeline Status Analyzer for Batched Simulations.

Analyzes which batch results are finished, in-progress, incomplete, or missing
in the results/ directory. Uses SLURM squeue to detect running/pending jobs.

Usage:
    python analyze_pipeline_status.py                # Full report on results/
    python analyze_pipeline_status.py --summary-only # Summary dashboard only
    python analyze_pipeline_status.py --json         # Machine-readable JSON
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

# =============================================================================
# Constants
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configs"
RESULTS_DIR = BASE_DIR / "results"
BATCH_FILE = BASE_DIR / "user_batches.json"

PLATFORMS = ["bluesky", "twitter", "reddit"]
PLATFORM_BATCHES = {"bluesky": 1, "twitter": 5, "reddit": 9}

SLOW_MODELS = ["70b", "apertus", "gemma"]

# Job names that belong to postprocessing (not simulation batch tracking)
POSTPROCESSING_JOB_NAMES = {
    "llm_judge", "llm_judge_resume", "llm_judge_resume3",
    "llm_judge_fix",
    "pp_judge", "pp_build_val", "pp_validate", "pp_features",
}

DIFFICULTY_LEVELS = [
    "base", "persona", "persona_style",
    "persona_style_context", "persona_style_context_finetuned"
]

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
ORANGE = "\033[38;5;208m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

N_RESPONSES_PER_USER = 20

SNAPSHOT_DIR = BASE_DIR / ".pipeline_snapshots"

# Map SLURM job name -> TSV task file (each row: config\tplatform\tbatch_id)
PHASE_TASK_FILES = {
    "phase1_fast":            BASE_DIR / "resubmit/tasks_phase1.tsv",
    "phase2_finetuned":       BASE_DIR / "resubmit/tasks_phase2.tsv",
    "phase3_llama70b":        BASE_DIR / "resubmit/tasks_phase3.tsv",
    "phase1_gemma_retry":     BASE_DIR / "resubmit/tasks_gemma_retry.tsv",
    "phase1_safety_quick":    BASE_DIR / "resubmit/tasks_phase1_safety_quick.tsv",
    "phase1_safety_standard": BASE_DIR / "resubmit/tasks_phase1_safety_standard.tsv",
    "phase2_finetuned_v2":    BASE_DIR / "resubmit/tasks_phase2_v2.tsv",
    "fill_incomplete":        BASE_DIR / "resubmit/tasks_fill_incomplete.tsv",
    "cuda_retry":             BASE_DIR / "resubmit/tasks_cuda_retry.tsv",
    # 12h resubmissions (same TSV files as originals)
    "phase2_ft_v2_12h":       BASE_DIR / "resubmit/tasks_phase2_v2.tsv",
    "phase3_llama_12h":       BASE_DIR / "resubmit/tasks_phase3.tsv",
    # backup jobs (depend on 12h, same TSV files)
    "phase2_ft_backup":       BASE_DIR / "resubmit/tasks_phase2_v2.tsv",
    "phase3_llama_backup":    BASE_DIR / "resubmit/tasks_phase3.tsv",
    # lowmem / 2gpu resubmissions (same TSV files)
    "phase2_ft_lowmem":       BASE_DIR / "resubmit/tasks_phase2_v2.tsv",
    "phase3_llama_2gpu":      BASE_DIR / "resubmit/tasks_phase3.tsv",
    # time-limit variants (same TSV file)
    "phase3_llama_2h":        BASE_DIR / "resubmit/tasks_phase3.tsv",
    "phase3_llama_4h":        BASE_DIR / "resubmit/tasks_phase3.tsv",
    "phase3_llama_24h":       BASE_DIR / "resubmit/tasks_phase3.tsv",
}


# =============================================================================
# Helper functions
# =============================================================================
def is_slow_config(config_name: str) -> bool:
    if "finetuned" in config_name:
        return True
    for model in SLOW_MODELS:
        if model in config_name.lower():
            return True
    return False


def config_to_slug(config: dict) -> str:
    """Convert a loaded config dict to the output filename slug.

    Must mirror the filename construction in run_simulation.py exactly.
    """
    parts = [
        config["model"].replace("/", "_"),
        "ft" if config.get("finetuned", False) else "noft",
        f"ctx{int(config.get('retrieve_context', False))}",
        f"style{config.get('n_style_examples', 0)}",
    ]
    slug = "__".join(parts)
    if not config.get("persona", True):
        slug += "__no_persona"
    if config.get("ft_variant"):
        slug += f"__{config['ft_variant']}"
    return slug


def load_configs() -> list:
    """Load all config files and return list of config info dicts."""
    configs = []
    for difficulty in DIFFICULTY_LEVELS:
        pattern = f"_{difficulty}.yaml"
        matching = sorted([
            f for f in os.listdir(CONFIG_DIR)
            if f.endswith(pattern)
        ])
        for cfg_name in matching:
            with open(CONFIG_DIR / cfg_name) as fh:
                config = yaml.safe_load(fh)
            config.setdefault("finetuned", False)
            config.setdefault("persona", True)
            config.setdefault("n_style_examples", 0)
            config.setdefault("retrieve_context", False)
            slug = config_to_slug(config)
            slow = is_slow_config(cfg_name)
            configs.append({
                "filename": cfg_name,
                "config": config,
                "slug": slug,
                "is_slow": slow,
            })
    return configs


def load_batch_info() -> dict:
    """Load user_batches.json and return structured info."""
    with open(BATCH_FILE) as f:
        data = json.load(f)
    info = {}
    for platform in PLATFORMS:
        pdata = data["platforms"][platform]
        batches = []
        for b in pdata["batches"]:
            batches.append({
                "batch_id": b["batch_id"],
                "n_users": b["n_users"],
                "users": b["users"],
                "expected_entries": b["n_users"] * N_RESPONSES_PER_USER,
            })
        info[platform] = {
            "total_users": pdata["total_users"],
            "n_batches": pdata["n_batches"],
            "batches": batches,
        }
    return info


def get_slurm_jobs() -> list:
    """Query SLURM for running/pending jobs with individual array task IDs.

    Returns a list of dicts with keys: job_id, array_task_id, name, state, time, time_limit, reason.
    Each array task element appears as a separate entry.
    """
    try:
        # SLURM_BITSTR_LEN prevents truncation of long array task ID specs
        env = os.environ.copy()
        env["SLURM_BITSTR_LEN"] = "1024"
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h",
             "-o", "%i %j %T %M %l %r"],
            capture_output=True, text=True, timeout=10, env=env
        )
        if result.returncode != 0:
            return []
        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            raw_id = parts[0]
            name = parts[1]
            state = parts[2]
            time_used = parts[3] if len(parts) > 3 else "?"
            time_limit = parts[4] if len(parts) > 4 else "?"
            reason = parts[5] if len(parts) > 5 else ""

            # Parse array job IDs like "957713_194" or "957713_[1-5,10%100]"
            if "_" in raw_id:
                base_job, task_part = raw_id.split("_", 1)
                task_ids = _expand_slurm_array_spec(task_part)
                for tid in task_ids:
                    jobs.append({
                        "job_id": base_job,
                        "array_task_id": tid,
                        "name": name,
                        "state": state,
                        "time": time_used,
                        "time_limit": time_limit,
                        "reason": reason,
                    })
            else:
                jobs.append({
                    "job_id": raw_id,
                    "array_task_id": None,
                    "name": name,
                    "state": state,
                    "time": time_used,
                    "time_limit": time_limit,
                    "reason": reason,
                })
        return jobs
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _expand_slurm_array_spec(spec: str) -> list:
    """Expand a SLURM array task spec like '[1-5,10,20-25%100]' or '194' into a list of ints.

    Handles the %N throttle suffix that SLURM appends to array specs (e.g. [1-5%100]).
    """
    spec = spec.strip("[]")
    # Remove %N throttle suffix (e.g. "1-5,10%100" -> "1-5,10")
    if "%" in spec:
        spec = spec[:spec.rfind("%")]
    if not spec:
        return []
    task_ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = part.split("-", 1)
                task_ids.extend(range(int(start), int(end) + 1))
            except ValueError:
                continue
        else:
            try:
                task_ids.append(int(part))
            except ValueError:
                continue
    return task_ids


def _load_task_file(tsv_path: Path) -> dict:
    """Load a resubmit TSV task file into {1-based task_id: (platform, config_filename, batch_id)}."""
    tasks = {}
    try:
        with open(tsv_path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                tasks[i] = (parts[1], parts[0], int(parts[2]))  # (platform, config, batch_id)
    except OSError:
        pass
    return tasks


def _job_name_to_phase(name: str) -> str | None:
    """Return the PHASE_TASK_FILES key that matches a SLURM job name, or None.

    Prefers longer (more specific) keys to avoid 'phase2_finetuned' matching
    'phase2_finetuned_v2'. A prefix match is only accepted if the next character
    is a non-alphanumeric separator (e.g. '_' followed by a job ID suffix).
    """
    best = None
    for phase_key in PHASE_TASK_FILES:
        if name == phase_key:
            return phase_key  # exact match wins immediately
        if name.startswith(phase_key):
            tail = name[len(phase_key):]
            if tail and not tail[0].isalnum():  # e.g. '_12345' suffix appended by SLURM
                if best is None or len(phase_key) > len(best):
                    best = phase_key
    return best


def build_slurm_task_map(jobs: list, configs: list) -> dict:
    """Map SLURM running/pending jobs to (platform, config_filename, batch_id).

    Uses the resubmit TSV files for direct task ID -> (platform, config, batch) lookup.

    Returns:
        dict: (platform, config_filename, batch_id) -> "RUNNING" or "PENDING"
    """
    # Load all phase TSV tables once
    phase_tables = {
        phase_key: _load_task_file(tsv_path)
        for phase_key, tsv_path in PHASE_TASK_FILES.items()
    }

    task_map = {}
    for job in jobs:
        if job["array_task_id"] is None:
            continue
        state = job["state"]
        if state not in ("RUNNING", "PENDING"):
            continue

        phase_key = _job_name_to_phase(job["name"])
        if phase_key is None:
            continue

        entry = phase_tables[phase_key].get(job["array_task_id"])
        if entry is None:
            continue

        key = entry  # (platform, config_filename, batch_id)
        # RUNNING takes priority over PENDING
        if key not in task_map or state == "RUNNING":
            task_map[key] = state

    return task_map


def load_json_safe(filepath: Path) -> list | None:
    """Load a JSON file, return None on error."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def colorize(status: str) -> str:
    """Color a status string."""
    s = status.upper()
    if s == "COMPLETE":
        return f"{GREEN}{status}{RESET}"
    elif s == "RUNNING":
        return f"{CYAN}{status}{RESET}"
    elif s == "PENDING":
        return f"{YELLOW}{status}{RESET}"
    elif s == "INCOMPLETE":
        return f"{ORANGE}{status}{RESET}"
    elif s == "NOT_STARTED":
        return f"{DIM}{status}{RESET}"
    return f"{YELLOW}{status}{RESET}"


# =============================================================================
# Batch Results Analysis (results/)
# =============================================================================
def compute_empirical_targets(configs, batch_info):
    """Compute the empirical target user count per (platform, batch).

    Because each user is only processed if they have >= 20 messages after
    the reproducible train-test split, the actual target per batch is lower
    than the raw count in user_batches.json. Since the split is deterministic,
    every config should produce the same user count for a given batch.

    We determine the target as the mode (most common user count) across all
    existing result files for that batch.
    """
    from collections import Counter

    targets = {}  # (platform, batch_id) -> target_users

    for platform in PLATFORMS:
        platform_dir = RESULTS_DIR / platform
        p_info = batch_info[platform]

        for batch_id in range(p_info["n_batches"]):
            user_counts = []
            for c in configs:
                batch_path = platform_dir / f"{c['slug']}__batch{batch_id}.json"
                if not batch_path.exists():
                    continue
                data = load_json_safe(batch_path)
                if data is not None and len(data) > 0:
                    n_users = len(set(e.get("user", "") for e in data))
                    user_counts.append(n_users)

            if user_counts:
                counter = Counter(user_counts)
                mode_users, mode_count = counter.most_common(1)[0]
                targets[(platform, batch_id)] = {
                    "target_users": mode_users,
                    "n_files_at_target": mode_count,
                    "n_files_total": len(user_counts),
                    "raw_users": p_info["batches"][batch_id]["n_users"],
                }
            else:
                # No files yet, fall back to raw count from user_batches.json
                targets[(platform, batch_id)] = {
                    "target_users": p_info["batches"][batch_id]["n_users"],
                    "n_files_at_target": 0,
                    "n_files_total": 0,
                    "raw_users": p_info["batches"][batch_id]["n_users"],
                }

    return targets


def analyze_batch_results(configs, batch_info, targets, slurm_task_map=None):
    """Analyze batch result files in results/.

    slurm_task_map is optional; if omitted, RUNNING/PENDING detection is skipped.

    For each config x platform x batch, determines status:
    - COMPLETE: file exists, user count >= empirical target
    - RUNNING: SLURM shows RUNNING for this (config, platform, batch)
    - PENDING: SLURM shows PENDING for this (config, platform, batch)
    - INCOMPLETE: file exists, users < target, no SLURM job for it
    - NOT_STARTED: no file, no SLURM job for it
    """
    report = {
        "platforms": {},
        "incomplete_jobs": [],
        "targets": targets,
    }

    for platform in PLATFORMS:
        platform_dir = RESULTS_DIR / platform
        p_info = batch_info[platform]
        expected_batch_ids = list(range(p_info["n_batches"]))

        platform_report = {
            "configs": {},
            "summary": {
                "complete": 0, "running": 0,
                "pending": 0, "incomplete": 0, "not_started": 0,
            },
            "batch_summary": {
                "total_expected": len(configs) * p_info["n_batches"],
                "complete": 0, "running": 0,
                "pending": 0, "incomplete": 0, "not_started": 0,
            },
        }

        for c in configs:
            slug = c["slug"]
            config_status = {
                "batches": {},
                "overall_status": "NOT_STARTED",
            }

            for batch_id in expected_batch_ids:
                batch_filename = f"{slug}__batch{batch_id}.json"
                batch_path = platform_dir / batch_filename

                has_file = batch_path.exists()
                target = targets[(platform, batch_id)]["target_users"]

                # Check SLURM for running/pending status
                slurm_state = slurm_task_map.get(
                    (platform, c["filename"], batch_id)
                ) if slurm_task_map is not None else None

                batch_status = {
                    "has_file": has_file,
                    "entries": 0,
                    "users": 0,
                    "target_users": target,
                    "raw_users": targets[(platform, batch_id)]["raw_users"],
                    "status": "NOT_STARTED",
                }

                if has_file:
                    data = load_json_safe(batch_path)
                    if data is not None:
                        batch_status["entries"] = len(data)
                        batch_status["users"] = len(set(
                            e.get("user", "") for e in data
                        ))

                        actual_u = batch_status["users"]

                        # Gemma configs: allow 1 missing user to still count as COMPLETE.
                        # Some users always produce [Response] headers (Gemma format
                        # artifact), leaving them with 0 valid entries. The fixed parser
                        # handles this, but legacy runs can be off by 1.
                        _is_gemma_cfg = "gemma" in c["filename"].lower()
                        _complete_threshold = max(target - 1, 0) if _is_gemma_cfg else target

                        if actual_u >= _complete_threshold:
                            batch_status["status"] = "COMPLETE"
                        elif slurm_state == "RUNNING":
                            batch_status["status"] = "RUNNING"
                        elif slurm_state == "PENDING":
                            batch_status["status"] = "PENDING"
                        else:
                            batch_status["status"] = "INCOMPLETE"
                            report["incomplete_jobs"].append({
                                "platform": platform,
                                "config": c["filename"],
                                "batch": batch_id,
                                "users": actual_u,
                                "target_users": target,
                                "entries": batch_status["entries"],
                                "file": str(batch_path),
                            })
                    else:
                        # JSON parse error — treat based on SLURM state
                        if slurm_state == "RUNNING":
                            batch_status["status"] = "RUNNING"
                        elif slurm_state == "PENDING":
                            batch_status["status"] = "PENDING"
                        else:
                            batch_status["status"] = "INCOMPLETE"
                            batch_status["error"] = "JSON parse error"
                else:
                    # No file
                    if slurm_state == "RUNNING":
                        batch_status["status"] = "RUNNING"
                    elif slurm_state == "PENDING":
                        batch_status["status"] = "PENDING"
                    # else: stays NOT_STARTED

                # Update batch-level summary
                status_key = batch_status["status"].lower()
                platform_report["batch_summary"][status_key] += 1

                config_status["batches"][batch_id] = batch_status

            # Determine overall config status
            statuses = [b["status"] for b in config_status["batches"].values()]
            has_incomplete = any(s == "INCOMPLETE" for s in statuses)
            has_running = any(s == "RUNNING" for s in statuses)
            has_pending = any(s == "PENDING" for s in statuses)

            if all(s == "COMPLETE" for s in statuses):
                config_status["overall_status"] = "COMPLETE"
                platform_report["summary"]["complete"] += 1
            elif all(s == "NOT_STARTED" for s in statuses):
                config_status["overall_status"] = "NOT_STARTED"
                platform_report["summary"]["not_started"] += 1
            elif has_running:
                config_status["overall_status"] = "RUNNING"
                platform_report["summary"]["running"] += 1
            elif has_pending:
                config_status["overall_status"] = "PENDING"
                platform_report["summary"]["pending"] += 1
            elif has_incomplete:
                config_status["overall_status"] = "INCOMPLETE"
                platform_report["summary"]["incomplete"] += 1
            else:
                # Mix of COMPLETE and NOT_STARTED (no running/pending/incomplete)
                config_status["overall_status"] = "INCOMPLETE"
                platform_report["summary"]["incomplete"] += 1

            platform_report["configs"][c["filename"]] = config_status

        report["platforms"][platform] = platform_report

    return report


def print_batch_results(report, configs, batch_info):
    targets = report["targets"]

    print(f"\n{BOLD}{'='*80}")
    print("BATCH RESULTS (results/)")
    print(f"{'='*80}{RESET}")

    for platform in PLATFORMS:
        pr = report["platforms"][platform]
        p_info = batch_info[platform]
        n_batches = p_info["n_batches"]
        s = pr["summary"]
        bs = pr["batch_summary"]

        # Show empirical targets
        target_users = [targets[(platform, b)]["target_users"] for b in range(n_batches)]
        raw_users = [targets[(platform, b)]["raw_users"] for b in range(n_batches)]

        print(f"\n{BOLD}--- {platform.upper()} ---{RESET}")
        print(f"  {DIM}Batches per config: {n_batches}")
        print(f"  Raw users per batch:    {raw_users}")
        print(f"  Target users per batch: {target_users}  (empirical, from completed runs){RESET}")
        print(f"  Configs: "
              f"{GREEN}{s['complete']} complete{RESET}, "
              f"{CYAN}{s['running']} running{RESET}, "
              f"{YELLOW}{s['pending']} pending{RESET}, "
              f"{ORANGE}{s['incomplete']} incomplete{RESET}, "
              f"{DIM}{s['not_started']} not started{RESET}")
        print(f"  Batches: "
              f"{GREEN}{bs['complete']}/{bs['total_expected']} complete{RESET}, "
              f"{CYAN}{bs['running']} running{RESET}, "
              f"{YELLOW}{bs['pending']} pending{RESET}, "
              f"{ORANGE}{bs['incomplete']} incomplete{RESET}, "
              f"{DIM}{bs['not_started']} not started{RESET}")

        # Print per-config details
        print()
        for c in configs:
            cs = pr["configs"][c["filename"]]
            status = cs["overall_status"]
            status_str = colorize(status)

            # Build batch bar
            bar_chars = []
            for b in range(n_batches):
                bs_item = cs["batches"].get(b, {})
                bstat = bs_item.get("status", "NOT_STARTED")
                if bstat == "COMPLETE":
                    bar_chars.append(f"{GREEN}#{RESET}")
                elif bstat == "RUNNING":
                    bar_chars.append(f"{CYAN}>{RESET}")
                elif bstat == "PENDING":
                    bar_chars.append(f"{YELLOW}P{RESET}")
                elif bstat == "INCOMPLETE":
                    bar_chars.append(f"{ORANGE}~{RESET}")
                else:
                    bar_chars.append(f"{DIM}.{RESET}")
            bar = "".join(bar_chars)

            n_complete = sum(
                1 for b in cs["batches"].values() if b["status"] == "COMPLETE"
            )

            short_name = c["filename"].replace(".yaml", "")
            speed = f"{YELLOW}S{RESET}" if c["is_slow"] else f"{GREEN}F{RESET}"

            print(f"  {speed} {short_name:<53} [{bar}] {n_complete}/{n_batches}  {status_str}")

            # Show details for non-complete, non-not_started batches
            if status in ("RUNNING", "PENDING", "INCOMPLETE"):
                for bid, bdata in sorted(cs["batches"].items()):
                    if bdata["status"] in ("RUNNING", "PENDING", "INCOMPLETE"):
                        if bdata["has_file"]:
                            print(f"      batch {bid}: "
                                  f"{bdata['users']}/{bdata['target_users']} users, "
                                  f"{bdata['entries']} entries  "
                                  f"{colorize(bdata['status'])}")
                        else:
                            print(f"      batch {bid}: no data file  "
                                  f"{colorize(bdata['status'])}")


# =============================================================================
# Incomplete Jobs Analysis
# =============================================================================
def print_incomplete_jobs(report):
    incomplete = report["incomplete_jobs"]
    if not incomplete:
        return

    print(f"\n{BOLD}{'='*80}")
    print(f"INCOMPLETE JOBS (partial results, not currently running or pending)")
    print(f"{'='*80}{RESET}")
    print(f"  {ORANGE}Found {len(incomplete)} incomplete batch jobs:{RESET}\n")

    # Group by platform
    by_platform = defaultdict(list)
    for c in incomplete:
        by_platform[c["platform"]].append(c)

    for platform in PLATFORMS:
        items = by_platform.get(platform, [])
        if not items:
            continue
        print(f"  {BOLD}{platform.upper()} ({len(items)} incomplete batches):{RESET}")
        for item in items:
            config_short = item["config"].replace(".yaml", "")
            pct = (item["users"] / item["target_users"] * 100) if item["target_users"] > 0 else 0
            print(f"    {config_short} batch {item['batch']}: "
                  f"{item['users']}/{item['target_users']} users ({pct:.0f}%), "
                  f"{item['entries']} entries")
        print()

    print(f"  {DIM}# Re-run run_simulation.py for each incomplete (platform, config, batch) above.{RESET}")


# =============================================================================
# SLURM Status
# =============================================================================
def print_slurm_status(jobs):
    print(f"\n{BOLD}{'='*80}")
    print("SLURM QUEUE STATUS")
    print(f"{'='*80}{RESET}")

    sim_jobs = [j for j in jobs if _job_name_to_phase(j["name"]) is not None]
    post_jobs = [j for j in jobs if j["name"] in POSTPROCESSING_JOB_NAMES]
    other_jobs = [j for j in jobs
                  if _job_name_to_phase(j["name"]) is None
                  and j["name"] not in POSTPROCESSING_JOB_NAMES]

    if not sim_jobs and not post_jobs and not other_jobs:
        print(f"  {DIM}No SLURM jobs found (squeue may not be available).{RESET}")
        return

    if sim_jobs:
        print(f"\n  Simulation jobs:")
        for phase_key in PHASE_TASK_FILES:
            phase_jobs = [j for j in sim_jobs if _job_name_to_phase(j["name"]) == phase_key]
            if not phase_jobs:
                continue
            running = len([j for j in phase_jobs if j["state"] == "RUNNING"])
            pending = len([j for j in phase_jobs if j["state"] == "PENDING"])
            # Show unique job IDs so the user can see if a retry is in the queue
            job_ids = sorted(set(j["job_id"] for j in phase_jobs))
            ids_str = f"{DIM}[{', '.join(job_ids)}]{RESET}"
            print(f"    {phase_key:<22} {GREEN}{running:>3} running{RESET}, "
                  f"{YELLOW}{pending:>4} pending{RESET}  {ids_str}")

    if post_jobs:
        print(f"\n  Postprocessing jobs:")
        by_name = defaultdict(list)
        for j in post_jobs:
            by_name[j["name"]].append(j)
        for name, jlist in sorted(by_name.items()):
            running = len([j for j in jlist if j["state"] == "RUNNING"])
            pending = len([j for j in jlist if j["state"] == "PENDING"])
            job_ids = sorted(set(j["job_id"] for j in jlist))
            ids_str = f"{DIM}[{', '.join(job_ids)}]{RESET}"
            print(f"    {name:<22} {GREEN}{running:>3} running{RESET}, "
                  f"{YELLOW}{pending:>4} pending{RESET}  {ids_str}")

    if other_jobs:
        print(f"\n  Other jobs: {len(other_jobs)}")
        for j in other_jobs[:5]:
            print(f"    {j['job_id']:>10} {j['name']:<20} {j['state']:<12} {j['time']}")


# =============================================================================
# Postprocessing Summary
# =============================================================================
def _postprocess_counts(cleaned_dir):
    """Count files at each postprocessing stage, per platform.

    Returns dict: platform → {judged, to_csv, val_built, bert_done, n_random}
    """
    stats = {}
    for platform in ["bluesky", "twitter", "reddit"]:
        pdir = cleaned_dir / platform
        if not pdir.exists():
            continue
        stats[platform] = {
            "n_random":  len(list(pdir.rglob("*_random_response.json"))),
            "judged":    len(list(pdir.rglob("*_optimal_response.json"))),
            "to_csv":    len(list(pdir.rglob("*_optimal_response.csv"))),
            "val_built": len(list(pdir.rglob("*_validation_data.csv"))),
            "bert_done": len(list(pdir.rglob("*_confusion_matrix.csv"))),
        }
    return stats


def _pp_cell(got, total, width=10):
    """Return a colored, fixed-width 'got/total' string for table columns."""
    if total == 0:
        text = "-"
        color = DIM
    else:
        text = f"{got}/{total}"
        color = GREEN if got == total else (CYAN if got > 0 else DIM)
    return f"{color}{text:<{width}}{RESET}"


def _print_postprocessing_summary(jobs):
    """Print full postprocessing pipeline status, per platform."""
    cleaned_dirs = sorted(
        list(BASE_DIR.glob("results_cleaned_*/")) +
        list(BASE_DIR.glob("results_2026_*/")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cleaned_dirs:
        return

    cleaned_dir = cleaned_dirs[0]
    stats = _postprocess_counts(cleaned_dir)
    if not stats:
        return

    platforms = [p for p in ["bluesky", "twitter", "reddit"] if p in stats]

    def _tot(key):
        return sum(s[key] for s in stats.values())

    total_random = _tot("n_random")
    total_judged = _tot("judged")
    total_csv    = _tot("to_csv")
    total_val    = _tot("val_built")
    total_bert   = _tot("bert_done")

    # Features: auc_results*.csv may be at root or in platform subdirs
    features_done = len(list(cleaned_dir.rglob("auc_results*.csv"))) > 0
    summary_done  = (cleaned_dir / "summary_metrics.csv").exists() or \
                    len(list(cleaned_dir.rglob("summary_metrics.csv"))) > 0

    # SLURM pp_validate job status
    val_jobs    = [j for j in jobs if j["name"] == "pp_validate"]
    val_running = sum(1 for j in val_jobs if j["state"] == "RUNNING")
    val_pending = sum(1 for j in val_jobs if j["state"] == "PENDING")

    print(f"\n  {BOLD}Postprocessing pipeline:{RESET}  {DIM}[{cleaned_dir.name}]{RESET}")

    # Header
    col_w = 10
    plat_hdrs = "  ".join(f"{p:<{col_w}}" for p in platforms)
    print(f"\n  {'Step':<24}  {plat_hdrs}  {'Total'}")
    print(f"  {'-'*70}")

    def _row(label, key_num, key_den, den_mult=1):
        cells = "  ".join(
            _pp_cell(stats[p][key_num], stats[p][key_den] * den_mult, col_w)
            for p in platforms
        )
        total_cell = _pp_cell(_tot(key_num), _tot(key_den) * den_mult, col_w)
        print(f"  {label:<24}  {cells}  {total_cell}")

    _row("LLM Judge",           "judged",    "n_random")
    _row("JSON → CSV",          "to_csv",    "judged")
    _row("Build val CSVs (×3)", "val_built", "judged",   den_mult=3)

    # BERT validate
    cells = "  ".join(
        _pp_cell(stats[p]['bert_done'], stats[p]['val_built'], col_w)
        for p in platforms
    )
    bert_total = _pp_cell(total_bert, total_val)
    bert_extra = ""
    if val_running:
        bert_extra += f"  {CYAN}{val_running} running{RESET}"
    if val_pending:
        bert_extra += f"  {YELLOW}{val_pending} pending{RESET}"
    print(f"  {'BERT validate':<24}  {cells}  {bert_total}{bert_extra}")

    # Features & post-process (no per-platform breakdown yet)
    feat_str = f"{GREEN}done{RESET}" if features_done else f"{DIM}pending{RESET}"
    summ_str = f"{GREEN}done{RESET}" if summary_done  else f"{DIM}pending{RESET}"
    print(f"  {'Features analysis':<24}  {feat_str}")
    print(f"  {'Post-process':<24}  {summ_str}")


# =============================================================================
# Summary Dashboard
# =============================================================================
def print_summary_dashboard(config_report, batch_report, jobs):
    print(f"\n{BOLD}{'='*80}")
    print("SUMMARY DASHBOARD")
    print(f"{'='*80}{RESET}")

    # SLURM
    sim_jobs = [j for j in jobs if _job_name_to_phase(j["name"]) is not None]
    sim_running = len([j for j in sim_jobs if j["state"] == "RUNNING"])
    sim_pending = len([j for j in sim_jobs if j["state"] == "PENDING"])

    print(f"\n  SLURM: {GREEN}{sim_running} running{RESET}, "
          f"{YELLOW}{sim_pending} pending{RESET}")
    n_incomplete = len(batch_report["incomplete_jobs"])
    if n_incomplete:
        print(f"  Incomplete jobs: {ORANGE}{n_incomplete}{RESET}")

    # Per-platform breakdown (config-level)
    print(f"\n  {BOLD}{'Platform':<12} {'Complete':>10} {'Running':>10} {'Pending':>10} {'Incomplete':>12} {'Not Started':>12} {'Total':>8}{RESET}")
    print(f"  {'-'*78}")

    totals = {"complete": 0, "running": 0, "pending": 0, "incomplete": 0, "not_started": 0, "total": 0}

    for platform in PLATFORMS:
        pr = batch_report["platforms"][platform]
        s = pr["summary"]
        total_p = len(pr["configs"])
        print(f"  {platform:<12} "
              f"{GREEN}{s['complete']:>10}{RESET} "
              f"{CYAN}{s['running']:>10}{RESET} "
              f"{YELLOW}{s['pending']:>10}{RESET} "
              f"{ORANGE}{s['incomplete']:>12}{RESET} "
              f"{DIM}{s['not_started']:>12}{RESET} "
              f"{total_p:>8}")
        for k in ("complete", "running", "pending", "incomplete", "not_started"):
            totals[k] += s[k]
        totals["total"] += total_p

    print(f"  {'-'*78}")
    print(f"  {'TOTAL':<12} "
          f"{GREEN}{totals['complete']:>10}{RESET} "
          f"{CYAN}{totals['running']:>10}{RESET} "
          f"{YELLOW}{totals['pending']:>10}{RESET} "
          f"{ORANGE}{totals['incomplete']:>12}{RESET} "
          f"{DIM}{totals['not_started']:>12}{RESET} "
          f"{totals['total']:>8}")

    # Batch-level breakdown
    print(f"\n  {BOLD}Batch-level progress:{RESET}")
    print(f"  {BOLD}{'Platform':<12} {'Complete':>10} {'Running':>10} {'Pending':>10} {'Incomplete':>12} {'Not Started':>12} {'Total':>8}{RESET}")
    print(f"  {'-'*78}")

    batch_totals = {"complete": 0, "running": 0, "pending": 0, "incomplete": 0, "not_started": 0, "total": 0}

    for platform in PLATFORMS:
        bs = batch_report["platforms"][platform]["batch_summary"]
        print(f"  {platform:<12} "
              f"{GREEN}{bs['complete']:>10}{RESET} "
              f"{CYAN}{bs['running']:>10}{RESET} "
              f"{YELLOW}{bs['pending']:>10}{RESET} "
              f"{ORANGE}{bs['incomplete']:>12}{RESET} "
              f"{DIM}{bs['not_started']:>12}{RESET} "
              f"{bs['total_expected']:>8}")
        for k in ("complete", "running", "pending", "incomplete", "not_started"):
            batch_totals[k] += bs[k]
        batch_totals["total"] += bs["total_expected"]

    print(f"  {'-'*78}")
    print(f"  {'TOTAL':<12} "
          f"{GREEN}{batch_totals['complete']:>10}{RESET} "
          f"{CYAN}{batch_totals['running']:>10}{RESET} "
          f"{YELLOW}{batch_totals['pending']:>10}{RESET} "
          f"{ORANGE}{batch_totals['incomplete']:>12}{RESET} "
          f"{DIM}{batch_totals['not_started']:>12}{RESET} "
          f"{batch_totals['total']:>8}")

    overall_pct = (batch_totals["complete"] / batch_totals["total"] * 100
                   if batch_totals["total"] > 0 else 0)
    print(f"\n  {BOLD}Overall batch completion: {overall_pct:.1f}%{RESET}")

    # Postprocessing status
    _print_postprocessing_summary(jobs)


# =============================================================================
# JSON Output
# =============================================================================
def generate_json_report(config_report, batch_report, jobs):
    targets = batch_report.get("targets", {})
    return {
        "config_matrix": config_report,
        "empirical_targets": {
            f"{p}_batch{b}": t
            for (p, b), t in sorted(targets.items())
        },
        "batch_results": {
            p: {
                "summary": batch_report["platforms"][p]["summary"],
                "batch_summary": batch_report["platforms"][p]["batch_summary"],
                "configs": {
                    k: {
                        "overall_status": v["overall_status"],
                        "batches": {
                            str(bid): {
                                "status": bdata["status"],
                                "users": bdata["users"],
                                "target_users": bdata["target_users"],
                                "entries": bdata["entries"],
                            }
                            for bid, bdata in v["batches"].items()
                        },
                    }
                    for k, v in batch_report["platforms"][p]["configs"].items()
                },
            }
            for p in PLATFORMS
        },
        "incomplete_jobs": batch_report["incomplete_jobs"],
        "slurm_jobs": jobs,
    }


# =============================================================================
# Config Matrix
# =============================================================================
def analyze_config_matrix(configs):
    fast = [c for c in configs if not c["is_slow"]]
    slow = [c for c in configs if c["is_slow"]]
    total_batches = sum(PLATFORM_BATCHES.values())

    return {
        "total_configs": len(configs),
        "fast_configs": len(fast),
        "slow_configs": len(slow),
        "total_batch_tasks": len(configs) * total_batches,
        "fast_tasks": len(fast) * total_batches,
        "slow_tasks": len(slow) * total_batches,
    }


def print_config_matrix(report, configs):
    print(f"\n{BOLD}{'='*80}")
    print("CONFIGURATION MATRIX")
    print(f"{'='*80}{RESET}")

    print(f"\nTotal configs: {report['total_configs']}")
    print(f"  Fast (6h): {report['fast_configs']} configs x 15 batches = {report['fast_tasks']} tasks")
    print(f"  Slow (24h): {report['slow_configs']} configs x 15 batches = {report['slow_tasks']} tasks")
    print(f"  Total batch tasks: {report['total_batch_tasks']}")

    print(f"\n  {'Config':<55} {'Type':<5}")
    print(f"  {'-'*65}")
    for c in configs:
        speed = f"{YELLOW}SLOW{RESET}" if c["is_slow"] else f"{GREEN}FAST{RESET}"
        short = c["filename"].replace(".yaml", "")
        print(f"  {short:<55} {speed}")


# =============================================================================
# Snapshot Storage & Change Tracking
# =============================================================================
def build_snapshot(batch_report):
    """Build a snapshot dict: (platform, config, batch) -> {status, entries, users}."""
    snapshot = {}
    for platform in PLATFORMS:
        pr = batch_report["platforms"][platform]
        for cfg_name, cs in pr["configs"].items():
            for batch_id, bdata in cs["batches"].items():
                key = f"{platform}/{cfg_name}/{batch_id}"
                snapshot[key] = {
                    "status": bdata["status"],
                    "entries": bdata["entries"],
                    "users": bdata["users"],
                }
    return snapshot


def save_snapshot(snapshot):
    """Save snapshot to .pipeline_snapshots/ with timestamp and latest copy."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_file = SNAPSHOT_DIR / f"snapshot_{timestamp}.json"
    latest_file = SNAPSHOT_DIR / "latest.json"

    data = {"timestamp": timestamp, "tasks": snapshot}
    with open(snapshot_file, "w") as f:
        json.dump(data, f, indent=2)

    # Update latest.json (copy, not symlink, for portability)
    with open(latest_file, "w") as f:
        json.dump(data, f, indent=2)


def load_previous_snapshot():
    """Load the previous latest.json snapshot if it exists."""
    latest_file = SNAPSHOT_DIR / "latest.json"
    if not latest_file.exists():
        return None
    try:
        with open(latest_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def compute_deltas(old_snapshot, new_snapshot):
    """Compare two snapshots and return a delta summary."""
    if old_snapshot is None:
        return None

    old_tasks = old_snapshot.get("tasks", {})
    new_tasks = new_snapshot

    new_completions = []
    progress_updates = []
    status_changes = []
    regressions = []

    status_rank = {"NOT_STARTED": 0, "INCOMPLETE": 1, "PENDING": 2, "RUNNING": 3, "COMPLETE": 4}

    all_keys = set(old_tasks.keys()) | set(new_tasks.keys())
    for key in sorted(all_keys):
        old = old_tasks.get(key, {"status": "NOT_STARTED", "entries": 0, "users": 0})
        new = new_tasks.get(key, {"status": "NOT_STARTED", "entries": 0, "users": 0})

        old_status = old["status"]
        new_status = new["status"]

        if old_status != new_status:
            if new_status == "COMPLETE" and old_status != "COMPLETE":
                new_completions.append({"key": key, "old_status": old_status})
            elif status_rank.get(new_status, 0) < status_rank.get(old_status, 0):
                regressions.append({"key": key, "old_status": old_status, "new_status": new_status})
            else:
                status_changes.append({"key": key, "old_status": old_status, "new_status": new_status})
        elif new["entries"] > old["entries"]:
            progress_updates.append({
                "key": key,
                "old_entries": old["entries"],
                "new_entries": new["entries"],
                "old_users": old["users"],
                "new_users": new["users"],
            })

    return {
        "new_completions": new_completions,
        "progress_updates": progress_updates,
        "status_changes": status_changes,
        "regressions": regressions,
        "old_timestamp": old_snapshot.get("timestamp", "unknown"),
    }


def print_deltas(deltas):
    """Print the CHANGES SINCE LAST RUN section."""
    if deltas is None:
        print(f"\n{DIM}  (No previous snapshot found — change tracking starts now){RESET}")
        return

    nc = len(deltas["new_completions"])
    pu = len(deltas["progress_updates"])
    sc = len(deltas["status_changes"])
    rg = len(deltas["regressions"])

    print(f"\n{BOLD}{'='*80}")
    print("CHANGES SINCE LAST RUN")
    print(f"{'='*80}{RESET}")
    print(f"  {DIM}Comparing against snapshot from {deltas['old_timestamp']}{RESET}")

    if nc == 0 and pu == 0 and sc == 0 and rg == 0:
        print(f"\n  {DIM}No changes detected.{RESET}")
        return

    print(f"\n  Summary: {GREEN}{nc} new completions{RESET}, "
          f"{CYAN}{pu} tasks progressed{RESET}, "
          f"{YELLOW}{sc} status changes{RESET}"
          + (f", {RED}{rg} regressions{RESET}" if rg else ""))

    if deltas["new_completions"]:
        print(f"\n  {GREEN}New completions:{RESET}")
        for item in deltas["new_completions"]:
            print(f"    {item['key']}  ({item['old_status']} -> COMPLETE)")

    if deltas["status_changes"]:
        print(f"\n  {YELLOW}Status changes:{RESET}")
        for item in deltas["status_changes"]:
            print(f"    {item['key']}  ({item['old_status']} -> {item['new_status']})")

    if deltas["regressions"]:
        print(f"\n  {RED}Regressions (unexpected):{RESET}")
        for item in deltas["regressions"]:
            print(f"    {item['key']}  ({item['old_status']} -> {item['new_status']})")

    if deltas["progress_updates"] and len(deltas["progress_updates"]) <= 20:
        print(f"\n  {CYAN}Progress updates:{RESET}")
        for item in deltas["progress_updates"]:
            print(f"    {item['key']}  "
                  f"entries: {item['old_entries']} -> {item['new_entries']}, "
                  f"users: {item['old_users']} -> {item['new_users']}")
    elif deltas["progress_updates"]:
        print(f"\n  {CYAN}Progress updates: {len(deltas['progress_updates'])} tasks progressed (too many to list){RESET}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze pipeline status for batched simulations"
    )
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only show summary dashboard")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    args = parser.parse_args()

    if args.no_color:
        global GREEN, YELLOW, RED, CYAN, MAGENTA, ORANGE, BOLD, DIM, RESET
        GREEN = YELLOW = RED = CYAN = MAGENTA = ORANGE = BOLD = DIM = RESET = ""

    # Load data
    print(f"{DIM}Loading configurations...{RESET}", file=sys.stderr)
    configs = load_configs()
    batch_info = load_batch_info()

    print(f"{DIM}Checking SLURM queue...{RESET}", file=sys.stderr)
    jobs = get_slurm_jobs()

    print(f"{DIM}Computing empirical targets...{RESET}", file=sys.stderr)
    config_report = analyze_config_matrix(configs)
    targets = compute_empirical_targets(configs, batch_info)

    print(f"{DIM}Mapping SLURM tasks to batches...{RESET}", file=sys.stderr)
    slurm_task_map = build_slurm_task_map(jobs, configs)

    print(f"{DIM}Analyzing batch results...{RESET}", file=sys.stderr)
    batch_report = analyze_batch_results(configs, batch_info, targets, slurm_task_map)

    # Snapshot & change tracking
    print(f"{DIM}Loading previous snapshot...{RESET}", file=sys.stderr)
    previous_snapshot = load_previous_snapshot()
    current_snapshot = build_snapshot(batch_report)
    deltas = compute_deltas(previous_snapshot, current_snapshot)
    save_snapshot(current_snapshot)

    # Output
    if args.json:
        json_report = generate_json_report(config_report, batch_report, jobs)
        print(json.dumps(json_report, indent=2, default=str))
        return

    print(f"\n{BOLD}{'#'*80}")
    print(f"  PIPELINE STATUS ANALYSIS  -  {len(configs)} configs x 3 platforms")
    print(f"{'#'*80}{RESET}")

    if not args.summary_only:
        print_config_matrix(config_report, configs)
        print_batch_results(batch_report, configs, batch_info)
        print_incomplete_jobs(batch_report)

    print_slurm_status(jobs)
    print_summary_dashboard(config_report, batch_report, jobs)
    print_deltas(deltas)

    print(f"\n{DIM}Legend: # = complete, > = running, P = pending, ~ = incomplete, . = not started")
    print(f"        F = fast (6h), S = slow (24h)")
    print(f"        Status detection: RUNNING/PENDING from SLURM squeue, not lock files")
    print(f"        Targets derived empirically (mode of user counts across completed runs){RESET}")


if __name__ == "__main__":
    main()
