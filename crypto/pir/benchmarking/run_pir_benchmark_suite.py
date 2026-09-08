#!/usr/bin/env python3
"""Runs PIR metrics benchmark across 1GB database configurations and prints formatted table."""

#
# To compile the binary with AVX512 and AVX2 hardware acceleration:
# blaze build -c opt --copt=-march=skylake-avx512 //privacy/private_membership/rlwe/v2/crypto/pir/benchmarking:pir_metrics_tool
#

import os
import subprocess
import sys
import time
from typing import Any, Dict

CONFIGS = [
    # ("1GB_Ell1_t4_Disk", 262144, 1, 4, True),
    # ("1GB_Ell1_t4_Mem", 262144, 1, 4, False),
    # ("1GB_Ell1_t8_Disk", 262144, 1, 8, True),
    # ("1GB_Ell1_t8_Mem", 262144, 1, 8, False),
    # ("1GB_Ell2_t2_Disk", 131072, 2, 2, True),
    # ("1GB_Ell2_t2_Mem", 131072, 2, 2, False),
    # ("1GB_Ell2_t4_Disk", 131072, 2, 4, True),
    # ("1GB_Ell2_t4_Mem", 131072, 2, 4, False),
    ("8GB_Ell4_t4_Mem", 2*262144, 4, 4, False),
    ("8GB_Ell4_t4_Disk", 2*262144, 4, 4, True),
    
]

version_tag = "v19"

LOG_FILE = f"privacy/private_membership/rlwe/v2/crypto/pir/benchmarking/logs/{version_tag}/pir_suite_metrics.log"
SUMMARY_FILE = f"privacy/private_membership/rlwe/v2/crypto/pir/benchmarking/logs/{version_tag}/pir_suite_summary.txt"
BINARY = (
    "privacy/private_membership/rlwe/v2/crypto/pir/benchmarking/pir_metrics_tool_bin_temp"
    if os.path.exists(
        "privacy/private_membership/rlwe/v2/crypto/pir/benchmarking/pir_metrics_tool_bin_temp"
    )
    else "blaze-bin/privacy/private_membership/rlwe/v2/crypto/pir/benchmarking/pir_metrics_tool"
)


def format_bytes(b):
  if b >= 1024**3:
    return f"{b / (1024**3):.2f} GB"
  elif b >= 1024**2:
    return f"{b / (1024**2):.2f} MB"
  elif b >= 1024:
    return f"{b / 1024:.2f} KB"
  return f"{b} B"


def format_time(ms):
  if ms >= 60000:
    return f"{ms / 60000:.2f} min"
  elif ms >= 1000:
    return f"{ms / 1000:.2f} s"
  return f"{ms:.2f} ms"


def format_count(c):
  return str(c)


metrics_map = [
    ("Number of entries (n / rows)", "num_entries", format_count),
    ("Entry size elements (ell*d / cols)", "entry_size_elements", format_count),
    ("Interpolation degree (t / slices)", "interpolation_degree", format_count),
    ("Switched Modulus q1 (bits)", "q1_bits", format_count),
    ("Switched Modulus q2 (bits)", "q2_bits", format_count),
    ("Verification Status", "verification_status", lambda x: str(x)),
    ("Offline Prep Time (Total)", "prep_server_time_ms", format_time),
    (
        "Offline Prep Time (Interpolate)",
        "prep_server_interp_time_ms",
        format_time,
    ),
    ("Offline Prep Time (Outputs)", "prep_server_outputs_time_ms", format_time),
    (
        "Offline Prep Time (Hint Matrix)",
        "prep_server_hints_time_ms",
        format_time,
    ),
    (
        "Offline Prep Time (Matrix Pack)",
        "prep_server_pack_time_ms",
        format_time,
    ),
    (
        "Server Preprocessed Size (Ideal)",
        "server_preprocessed_ideal_bytes",
        format_bytes,
    ),
    (
        "Server Stored Size (Disk)",
        "server_disk_storage_bytes",
        format_bytes,
    ),
    ("Online Query Size (Ideal)", "c2s_on_msg_bytes_ideal", format_bytes),
    (
        "Online Response Size (Ideal)",
        "s2c_on_msg_bytes_ideal",
        format_bytes,
    ),
    ("Online Query Gen Time (Client)", "client_query_gen_time_ms", format_time),
    (
        "Server Response Latency (Total)",
        "server_response_latency_ms",
        format_time,
    ),
    (
        "Server Data Load Time",
        "server_data_load_time_ms",
        format_time,
    ),
    (
        "Online Response Gen Time (Proc)",
        "server_proc_resp_time_ms",
        format_time,
    ),
    (
        "Online Response Gen Time (DB Mult)",
        "server_proc_resp_db_mult_time_ms",
        format_time,
    ),
    (
        "Online Response Gen Time (Packing)",
        "server_proc_resp_packing_time_ms",
        format_time,
    ),
    (
        "Online Response Gen Time (Poly Eval)",
        "server_proc_resp_poly_eval_time_ms",
        format_time,
    ),
    (
        "Online Response Dec Time (Client)",
        "client_proc_resp_time_ms",
        format_time,
    ),
]


def main():
  os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
  print(f"Starting PIR Benchmark Suite for {len(CONFIGS)} configurations...")

  completed_labels = set()
  if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
      for line in f:
        if line.startswith("=== BENCHMARK_RESULT:"):
          lbl = line.split("=== BENCHMARK_RESULT: ")[1].split(" ===")[0].strip()
          completed_labels.add(lbl)

  print(f"Skipping {len(completed_labels)} already completed configurations.")

  temp_log_files = []
  for item in CONFIGS:
    label, n, ell, t, use_disk = item
    if label in completed_labels:
      print(f"Skipping {label} (already done).")
      continue
    temp_log = f"{LOG_FILE}.{label}"
    temp_log_files.append((label, temp_log))
    disk_str = "Disk Storage" if use_disk else "In-Memory"
    print(
        f"---> Starting {label} ({disk_str}, n={n}, ell={ell}, t={t})"
        " sequentially...",
        flush=True,
    )
    cmd = [
        BINARY,
        f"--num_entries={n}",
        f"--entry_size_multiple={ell}",
        f"--interpolation_degree={t}",
        f"--use_disk_storage={'true' if use_disk else 'false'}",
        f"--label={label}",
        f"--output_file={temp_log}",
        "--iterations=3",
        "--pack_log_digit=19",
        "--pack_num_digits=2",
        "--skip_verification=false",
    ]
    start_t = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - start_t
    if proc.returncode != 0:
      print(f"ERROR running {label} (exit {proc.returncode})", flush=True)
      if proc.stdout is not None:
        print(proc.stdout)
    else:
      print(f"Finished {label} in {elapsed:.1f}s.", flush=True)

  # Concatenate temp logs to LOG_FILE
  with open(LOG_FILE, "a") as main_log:
    for _, temp_log in temp_log_files:
      if os.path.exists(temp_log):
        with open(temp_log, "r") as tl:
          main_log.write(tl.read())

  # Parse LOG_FILE and generate summary table
  if not os.path.exists(LOG_FILE):
    print("No log file produced.")
    return

  results = []
  cur_res: Dict[str, Any] = {}
  with open(LOG_FILE, "r") as f:
    for line in f:
      line = line.strip()
      if line.startswith("=== BENCHMARK_RESULT:"):
        cur_res = {
            "label": line.split("=== BENCHMARK_RESULT: ")[1].split(" ===")[0]
        }
      elif line.startswith("========================================"):
        if cur_res:
          # Compute entry_size_elements
          if "entry_size_multiple" in cur_res:
            cur_res["entry_size_elements"] = (
                cur_res["entry_size_multiple"] * 2048
            )
          results.append(cur_res)
        cur_res = {}
      elif "=" in line and cur_res is not None:
        k, v = line.split("=", 1)
        try:
          cur_res[k] = float(v) if "." in v else int(v)
        except ValueError:
          cur_res[k] = v

  # Build ASCII/Markdown Tables per DB Size
  db_sizes = ["64MB", "1GB", "2GB", "4GB", "8GB"]
  out_lines = []

  for db in db_sizes:
    db_results = [r for r in results if str(r.get("label", "")).startswith(db)]
    if not db_results:
      continue

    headers = [
        "Metric / Configuration",
        "ell=1 t=4 (Disk)",
        "ell=1 t=4 (Mem)",
        "ell=1 t=8 (Disk)",
        "ell=1 t=8 (Mem)",
        "ell=2 t=2 (Disk)",
        "ell=2 t=2 (Mem)",
        "ell=2 t=4 (Disk)",
        "ell=2 t=4 (Mem)",
    ]

    out_lines.append("\n" + "=" * 191)
    out_lines.append(f"PIR BENCHMARK SUITE SUMMARY TABLE ({db} Database)")
    out_lines.append("=" * 191)

    row_fmt = (
        "{:<36} | {:<16} | {:<16} | {:<16} | {:<16} | {:<16} | {:<16} | {:<16}"
        " | {:<16}"
    )
    out_lines.append(row_fmt.format(*headers))
    out_lines.append("-" * 191)

    configs_to_match = [
        (1, 4, True),
        (1, 4, False),
        (1, 8, True),
        (1, 8, False),
        (2, 2, True),
        (2, 2, False),
        (2, 4, True),
        (2, 4, False),
    ]
    for name, key, fmt_fn in metrics_map:
      row_vals = [name]
      for ell_val, t_val, disk_val in configs_to_match:
        matched = [
            r
            for r in db_results
            if r.get("entry_size_multiple") == ell_val
            and r.get("interpolation_degree") == t_val
            and (
                str(r.get("use_disk_storage", "")).lower()
                == ("true" if disk_val else "false")
                or r.get("use_disk_storage") == disk_val
            )
        ]
        if matched:
          row_vals.append(fmt_fn(matched[0].get(key, 0)))
        else:
          row_vals.append("N/A")
      out_lines.append(row_fmt.format(*row_vals))
      if key in [
          "verification_status",
          "prep_server_pack_time_ms",
          "server_disk_storage_bytes",
          "s2c_on_msg_bytes_ideal",
      ]:
        out_lines.append("-" * 191)

    out_lines.append("=" * 191)

  full_txt = "\n".join(out_lines) + "\n"
  print(full_txt)
  with open(SUMMARY_FILE, "w") as f:
    f.write(full_txt)
  print(f"Summary table written to {SUMMARY_FILE} and full logs in {LOG_FILE}.")


if __name__ == "__main__":
  main()
