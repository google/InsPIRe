#!/usr/bin/env python3
"""Runs VPIR metrics benchmark across 1GB, 2GB, and 4GB database sizes and prints formatted table."""

#
# To compile the binary with AVX512 and AVX2 hardware acceleration:
# blaze build -c opt --copt=-march=skylake-avx512 //privacy/private_membership/rlwe/v2/crypto/vpir/benchmarking:vpir_metrics_tool
#

import os
import subprocess
import sys
import time
from typing import Any, Dict

BASE_CONFIGS = [
    # ("64MB_Ell16_n2048", 2048, 16),
    ("1GB_Ell16_n32768", 32768, 16),
    ("1GB_Ell8_n65536", 65536, 8),
    # ("1GB_Ell4_n131072", 131072, 4),
    # ("1GB_Ell2_n262144", 262144, 2),
]

CONFIGS = []
for base_label, n, ell in BASE_CONFIGS:
  CONFIGS.append((f"{base_label}_standard", n, ell, False))
  CONFIGS.append((f"{base_label}_local_finalize", n, ell, True))

version_tag = "v20"

LOG_FILE = f"privacy/private_membership/rlwe/v2/crypto/vpir/benchmarking/logs/{version_tag}/vpir_suite_metrics.log"
SUMMARY_FILE = f"privacy/private_membership/rlwe/v2/crypto/vpir/benchmarking/logs/{version_tag}/vpir_suite_summary.txt"
BINARY = "blaze-bin/privacy/private_membership/rlwe/v2/crypto/vpir/benchmarking/vpir_metrics_tool"


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
    ("Local Finalize Enabled", "local_finalize", format_count),
    ("Number of entries (n / rows)", "num_entries", format_count),
    ("Entry size elements (ell*d / cols)", "entry_size_elements", format_count),
    ("Verification Status", "verification_status", format_count),
    ("Verification Mismatches", "verification_mismatches", format_count),
    ("Offline Prep Time (Server)", "prep_server_time_ms", format_time),
    (
        "Offline Proof Gen Time (Server)",
        "off_server_proof_time_ms",
        format_time,
    ),
    ("Offline Process Time (Client)", "off_client_time_ms", format_time),
    (
        "Offline C2S Msg Size (Formula)",
        "c2s_off_msg_bytes_formula",
        format_bytes,
    ),
    (
        "Offline S2C Static Hint Size (Formula)",
        "s2c_off_hint_msg_bytes_formula",
        format_bytes,
    ),
    (
        "Offline S2C Dynamic Proof Size (Formula)",
        "s2c_off_proof_msg_bytes_formula",
        format_bytes,
    ),
    (
        "Offline S2C Total Msg Size (Formula)",
        "s2c_off_msg_bytes_formula",
        format_bytes,
    ),
    (
        "Client Stored Size (Formula)",
        "client_store_bytes_formula",
        format_bytes,
    ),
    (
        "Server Stored Size (Formula)",
        "server_store_bytes_formula",
        format_bytes,
    ),
    ("Online Query Size (Formula)", "c2s_on_msg_bytes_formula", format_bytes),
    (
        "Online Response Size (Formula)",
        "s2c_on_msg_bytes_formula",
        format_bytes,
    ),
    ("Online Query Gen Time (Client)", "client_query_gen_time_ms", format_time),
    (
        "Online Response Gen Time (Server)",
        "server_proc_resp_time_ms",
        format_time,
    ),
    (
        "Online Response Verify Time (Client)",
        "client_resp_verify_time_ms",
        format_time,
    ),
    (
        "Online Response Decrypt Time (Client)",
        "client_resp_decrypt_time_ms",
        format_time,
    ),
    (
        "Online Response Total Dec Time (Client)",
        "client_proc_resp_time_ms",
        format_time,
    ),
]


def main():
  os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

  completed_labels = set()
  if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
      for line in f:
        if line.startswith("=== BENCHMARK_RESULT:"):
          lbl = line.split("=== BENCHMARK_RESULT: ")[1].split(" ===")[0].strip()
          completed_labels.add(lbl)

  print(f"Starting VPIR Benchmark Suite (v20) for {len(CONFIGS)} configurations...")
  print(f"Skipping {len(completed_labels)} already completed configurations.")

  temp_log_files = []

  for label, n, ell, local_finalize in CONFIGS:
    if label in completed_labels:
      continue
    temp_log = f"{LOG_FILE}.{label}"
    temp_log_files.append((label, temp_log))
    print(
        f"---> Starting {label} (n={n}, ell={ell}, local_finalize={local_finalize}) sequentially...",
        flush=True,
    )
    cmd = [
        BINARY,
        f"--num_entries={n}",
        f"--entry_size_multiple={ell}",
        f"--label={label}",
        f"--output_file={temp_log}",
        "--iterations=3",
        f"--local_finalize={str(local_finalize).lower()}",
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
        "ell=16 (Std)",
        "ell=16 (LocFin)",
        "ell=8 (Std)",
        "ell=8 (LocFin)",
        "ell=4 (Std)",
        "ell=4 (LocFin)",
        "ell=2 (Std)",
        "ell=2 (LocFin)",
    ]

    out_lines.append("\n" + "=" * 175)
    out_lines.append(
        f"VPIR BENCHMARK SUITE SUMMARY TABLE ({db} Database - Standard vs Local Finalize)"
    )
    out_lines.append("=" * 175)

    row_fmt = "{:<40} | {:<14} | {:<14} | {:<14} | {:<14} | {:<14} | {:<14} | {:<14} | {:<14}"
    out_lines.append(row_fmt.format(*headers))
    out_lines.append("-" * 175)

    columns = [
        (16, 0),
        (16, 1),
        (8, 0),
        (8, 1),
        (4, 0),
        (4, 1),
        (2, 0),
        (2, 1),
    ]

    for name, key, fmt_fn in metrics_map:
      row_vals = [name]
      for ell_val, loc_fin in columns:
        matched = [
            r
            for r in db_results
            if r.get("entry_size_multiple") == ell_val
            and int(r.get("local_finalize", 0)) == loc_fin
        ]
        if matched:
          row_vals.append(fmt_fn(matched[0].get(key, 0)))
        else:
          row_vals.append("N/A")
      out_lines.append(row_fmt.format(*row_vals))

    out_lines.append("=" * 175 + "\n")

  full_txt = "\n".join(out_lines)
  print(full_txt)
  with open(SUMMARY_FILE, "w") as f:
    f.write(full_txt)
  print(f"Summary table written to {SUMMARY_FILE} and full logs in {LOG_FILE}.")


if __name__ == "__main__":
  main()
