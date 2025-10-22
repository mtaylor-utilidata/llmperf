import argparse
import csv
import json
import math
import os
import random
import re
import threading
import time
import requests
import boto3
import logging
import sys
import statistics
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
from queue import Queue

import pandas as pd
import ray
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizerFast


from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients

from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    randomly_sample_sonnet_lines_prompt,
    build_scheduled_sonnet_prompt,
    LLMPerfResults,
    sample_random_positive_int,
)

def run_schedule_mode(
        *,
        llm_api: str,
        model: str,
        schedule_file: str,
        results_dir: str,
        additional_sampling_params: str,
        max_sampled_requests_per_second: int = 15,
        use_subdir: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Dispatch requests using a schedule file with delta timestamps.
    Preserves concurrency by launching requests in background threads at their appropriate offsets.
    Samples requests per second to cap throughput.
    """
    # Determine whether to use a subdirectory for results
    if use_subdir:
        results_subdir_path = _prepare_results_subdir(results_dir, schedule_file)
    else:
        results_subdir_path = Path(results_dir)

    # Copy schedule file to results directory for reference
    copyfile(schedule_file, results_subdir_path / "schedule.csv")

    # Create output directory and copy schedule file
    log_fh = open(results_subdir_path / "requests_sent.log", "a")

    # Load tokenizer
    tokenizer = get_tokenizer_for_model(model)
    get_token_length = lambda text: len(tokenizer.encode(text))

    # Read and parse the schedule CSV
    schedule = _load_schedule(schedule_file)

    # --- NEW: apply sampling before any threads are allocated ---
    schedule_sampled, schedule_not_sampled = _sample_schedule_records(schedule, max_sampled_requests_per_second)

    # Build a pool of launchers
    num_launchers = min(700, len(schedule_sampled))
    launcher_pool = Queue(maxsize=num_launchers)
    for _ in range(num_launchers):
        clients = construct_clients(llm_api=llm_api, num_clients=1)
        launcher_pool.put(RequestsLauncher(clients))

    delay = 5
    start_time_mono = time.monotonic() + delay
    launch_time = datetime.fromtimestamp(time.time() + delay, tz=timezone.utc)
    max_response_log_str = f"  Schedule should continue on track as long as requests don't exceed ~{num_launchers / max_sampled_requests_per_second}s e2e latency on average." if num_launchers < len(schedule_sampled) else ""

    logger.info(
        f"\n*** Scheduled launch starting in {delay}s at {launch_time.isoformat(timespec='seconds')} *** \n"
        f"  Total sampled requests to launch: {len(schedule_sampled)} \n"
        f"  Max number of requests measured per second: {max_sampled_requests_per_second} \n"
        f"  Number of request launchers in pool: {num_launchers} \n"
        f"{max_response_log_str}\n"
    )
    t0_utc = time.time()

    # Thread-safe collection of completed request metrics
    completed_sampled_requests = []
    completed_sampled_lock = threading.Lock()


    # Logging and progress tracking
    unsampled_counter = {"count": 0}
    unsampled_lock = threading.Lock()

    dispatch_stats_sampled = {"lags": []}
    dispatch_stats_sampled_lock = threading.Lock()
    dispatch_stats_unsampled = {"lags": []}
    dispatch_stats_unsampled_lock = threading.Lock()

    log_lock = threading.Lock()
    executor_sampled = ThreadPoolExecutor(max_workers=750)
    executor_unsampled = ThreadPoolExecutor(max_workers=200)

    # --- Run sampled schedule ---
    for sched in schedule_sampled:
        executor_sampled.submit(
            _launch_and_record_scheduled,
            sched,
            start_time_mono,
            t0_utc,
            model,
            llm_api,
            additional_sampling_params,
            tokenizer,
            get_token_length,
            completed_sampled_requests,
            completed_sampled_lock,
            log_lock,
            log_fh,
            launcher_pool,
            dispatch_stats_sampled,
            dispatch_stats_sampled_lock,
        )

    # --- Run unsampled schedule concurrently ---
    for sched in schedule_not_sampled:
        executor_unsampled.submit(
            _run_unsampled_records,
            sched,
            start_time_mono,
            t0_utc,
            model,
            llm_api,
            additional_sampling_params,
            tokenizer,
            get_token_length,
            log_lock,
            log_fh,
            unsampled_counter,
            unsampled_lock,
            dispatch_stats_unsampled,
            dispatch_stats_unsampled_lock,
        )

    progress_thread = threading.Thread(
        target=_log_progress_periodically,
        args=(
            completed_sampled_requests,
            completed_sampled_lock,
            len(schedule_sampled),
            unsampled_counter,
            unsampled_lock,
            len(schedule_not_sampled),
            min(max_sampled_requests_per_second, len(schedule_sampled)),
            start_time_mono,
            dispatch_stats_sampled,
            dispatch_stats_unsampled,
            dispatch_stats_unsampled_lock,
            dispatch_stats_sampled_lock,
            launcher_pool
        ),
        daemon=True,
    )
    progress_thread.start()

    # --- Wait for both to complete ---
    executor_sampled.shutdown(wait=True)
    executor_unsampled.shutdown(wait=True)
    progress_thread.join(timeout=2.0)  # allow it to exit cleanly
    log_fh.close()

    time.sleep(2)  # ensure all logs are flushed

    logger.info(f"\nResults for schedule-mode benchmark for {model} queried with {llm_api} API.\n")

    summary = metrics_summary(completed_sampled_requests, start_time_mono, time.monotonic())
    summary.update({
        "model": model,
        "num_concurrent_requests": "scheduled",
        "num_launched": len(schedule_sampled),
        "num_unsampled": len(schedule_not_sampled),
        "schedule_file": schedule_file,
        "results_subdir": str(results_subdir_path),
        "wall_time_s": time.monotonic() - start_time_mono,
    })

    return summary, completed_sampled_requests

def _log_progress_periodically(
        completed_requests,
        completed_lock,
        sampled_total,
        unsampled_counter,
        unsampled_lock,
        unsampled_total,
        max_sampled_requests_per_second,
        start_time_mono=None,
        dispatch_stats_sampled=None,
        dispatch_stats_unsampled=None,
        dispatch_stats_unsampled_lock=None,
        dispatch_stats_sampled_lock=None,
        launcher_pool=None,
):
    """Logs progress and dispatch timing stats every few seconds."""

    interval_s=10.0

    # --- Wait until launch start ---
    if start_time_mono:
        delay = max(0, start_time_mono - time.monotonic())
        if delay > 0:
            time.sleep(delay)

    iteration = 0
    while True:
        iteration += 1
        time.sleep(interval_s)
        now = time.monotonic()
        elapsed = (now - start_time_mono) if start_time_mono else 0.0

        # --- Count progress ---
        with completed_lock:
            sampled_done = len(completed_requests)
        with unsampled_lock:
            unsampled_done = unsampled_counter["count"]

        total_done = sampled_done + unsampled_done
        total_all = sampled_total + unsampled_total

        # --- Aggregate dispatch lag stats ---
        def _stats(lags):
            if not lags:
                return (0.0, 0.0, 0.0)
            return (
                statistics.mean(lags),
                max(lags),
                statistics.median(lags),
            )

        with dispatch_stats_sampled_lock:
            mean_s, max_s, med_s = _stats(dispatch_stats_sampled["lags"])
        with dispatch_stats_unsampled_lock:
            mean_u, max_u, med_u = _stats(dispatch_stats_unsampled["lags"])

        # --- Launcher pool saturation ---
        saturation_pct = None
        if launcher_pool is not None:
            total_launchers = launcher_pool.maxsize
            idle_launchers = launcher_pool.qsize()
            busy_launchers = total_launchers - idle_launchers
            saturation_pct = (busy_launchers / total_launchers * 100.0) if total_launchers else 0.0
            pool_status = f" | Launchers busy: {busy_launchers}/{total_launchers} ({saturation_pct:.1f}% used)"
        else:
            pool_status = " | (no launcher pool info)"

        #Calculate response timings
        response_timings = ""
        if len(completed_requests) > 0: # Average the end to end latency across the last 16 completed requests
            # you can get the e2e from an element of completed requests with element.get(common_metrics.E2E_LAT, 0)
            number_to_average = int(max_sampled_requests_per_second * interval_s)
            recent_latencies = [r.get(common_metrics.E2E_LAT, 0) for r in completed_requests[-number_to_average:]]
            avg_recent_latency = sum(recent_latencies) / len(recent_latencies)
            response_timings = f" | Average e2e latency of last {len(recent_latencies)} sampled requests: {avg_recent_latency:.3f}s"

        # --- Emit progress line ---
        logger.info(
            f"[progress +{elapsed:.1f}s] "
            f"Sampled: {sampled_done}/{sampled_total} | "
            f"Unsampled: {unsampled_done}/{unsampled_total} | "
            f"Total: {total_done}/{total_all}"
            f"{pool_status}"
            f"{response_timings}"
        )

        #warn if we are running out of launchers (over 95% saturation)
        if launcher_pool is not None and saturation_pct is not None and saturation_pct > 95.0 and saturation_pct < 99.99:
            logger.warning(f"High launcher pool saturation: {saturation_pct:.1f}% used. Consider lowering '--max-sampled-requests-per-second'.")
        elif launcher_pool is not None and saturation_pct is not None and saturation_pct >= 99.99:
            logger.error(f"Critical launcher pool saturation: {saturation_pct:.1f}% used. Upcoming requests will likely be delayed from their scheduled dispatch. Consider lowering '--max-sampled-requests-per-second'.")

        #if the modulo of iteration is zero, log the lag stats
        if iteration % 3 == 0:
            _log_lag_statistics(max_s, max_u, mean_s, mean_u, med_s, med_u)

        if total_done >= total_all:
            _log_lag_statistics(max_s, max_u, mean_s, mean_u, med_s, med_u)
            if max_s > 1 or max_u > 1:
                logger.error("Some requests had significant dispatch lag. Check logs for details.")
            elif max_s > .05 or max_u > .05:
                logger.warn("Some requests had more dispatch lag than expected. Check logs for details.")
            break


def _log_lag_statistics(max_s, max_u, mean_s, mean_u, med_s, med_u):
    logger.info(
        f"  Lag stats — "
        f"Sampled: avg {mean_s:.3f}s | med {med_s:.3f}s | max {max_s:.3f}s || "
        f"Unsampled: avg {mean_u:.3f}s | med {med_u:.3f}s | max {max_u:.3f}s"
    )


def _sample_schedule_records(
        schedule: List[Dict[str, Any]],
        max_sampled_requests_per_second: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Bins schedule records into 1-second intervals and deterministically samples
    up to `max_sampled_requests_per_second` per bin. Within each bin, samples
    are evenly spaced across available offsets.
    """
    import math

    if not schedule:
        return [], []

    # Ensure order by scheduled_offset_s
    schedule = sorted(schedule, key=lambda x: x["scheduled_offset_s"])

    # Group into integer-second bins
    bins: Dict[int, List[Dict[str, Any]]] = {}
    for rec in schedule:
        bin_idx = int(math.floor(rec["scheduled_offset_s"]))
        bins.setdefault(bin_idx, []).append(rec)
        rec["sample"] = False

    # Deterministic even selection
    for bin_idx, records in bins.items():
        n = len(records)
        to_sample = min(max_sampled_requests_per_second, n)

        if to_sample == 0:
            continue

        if to_sample >= n:
            selected_indices = list(range(n))
        else:
            # Evenly distributed indices across [0, n-1]
            step = (n - 1) / (to_sample - 1) if to_sample > 1 else 0
            selected_indices = [int(round(i * step)) for i in range(to_sample)]

        for idx in selected_indices:
            records[idx]["sample"] = True

        logger.debug(
            f"[Bin {bin_idx:02d}s] Sampled {len(selected_indices)} of {n} records "
            f"at indices {selected_indices}"
        )

    # Split out sampled vs unsampled
    schedule_sampled = [r for r in schedule if r["sample"]]
    schedule_not_sampled = [r for r in schedule if not r["sample"]]

    logger.info(
        f"***Sample Sizes Calculated *** Total sampled: {len(schedule_sampled)}, "
        f"not sampled: {len(schedule_not_sampled)}"
    )

    return schedule_sampled, schedule_not_sampled


def _prepare_results_subdir(results_dir: str, schedule_file: str) -> Path:
    """Creates a timestamped subdirectory"""
    utc_time = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    subdir_name = f"{utc_time}_schedule_run"
    results_subdir_path = Path(results_dir) / subdir_name
    results_subdir_path.mkdir(parents=True, exist_ok=True)
    return results_subdir_path

def _load_schedule(schedule_file: str) -> List[Dict[str, Any]]:
    """Loads and parses the schedule file into a list of request dicts."""
    schedule = []
    with open(schedule_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            schedule.append({
                "request_id": idx + 1,
                "scheduled_offset_s": float(row["scheduled_offset_s"]),
                "input_tokens": int(row["input_tokens"]),
                "output_tokens": int(row["output_tokens"]),
            })
    return schedule

def _launch_and_record_scheduled(
        sched: Dict[str, Any],
        start_time_mono: float,
        t0_utc: float,
        model: str,
        llm_api: str,
        additional_sampling_params: str,
        tokenizer: AutoTokenizer,
        get_token_length,
        completed_sampled_requests: List[Dict[str, Any]],
        completed_sampled_lock: threading.Lock,
        log_lock: threading.Lock,
        log_fh,
        launcher_pool,   # <-- pool of RequestsLauncher
        dispatch_stats_sampled: Dict[str, List[float]],
        dispatch_stats_sampled_lock: threading.Lock,
):
    request_id = sched["request_id"]
    scheduled_offset = sched["scheduled_offset_s"]

    # Sleep until ~10s before scheduled time to prep work
    time.sleep(max(0, start_time_mono + scheduled_offset - time.monotonic() - .1))  #.1s prep buffer
    prep_start = time.monotonic()


    # Prepare request inputs and configs
    prompt = build_scheduled_sonnet_prompt(
        input_tokens=sched["input_tokens"],
        output_tokens=sched["output_tokens"],
        tokenizer=tokenizer
    )
    sampling_params = {"max_tokens": sched["output_tokens"]}
    sampling_params.update(json.loads(additional_sampling_params))

    request_config = RequestConfig(
        model=model,
        prompt=prompt,
        sampling_params=sampling_params,
        llm_api=llm_api,
    )

    prep_end = time.monotonic()
    logger.debug(f"[request #{request_id}] Request prep completed with duration {prep_end - prep_start:.3f}s")

    # Final sleep until scheduled time
    time.sleep(max(0, start_time_mono + scheduled_offset - time.monotonic()))

    # --- Acquire a launcher from the pool ---
    req_launcher = launcher_pool.get()

    launcher_released = False
    try:
        # Launch and record dispatch
        req_launcher.launch_requests(request_config)
        dispatch_ts_mono = time.monotonic()
        dispatch_offset = dispatch_ts_mono - start_time_mono
        dispatch_lag = dispatch_offset - scheduled_offset
        dispatch_ts = time.time()
        dispatch_ts_utc = datetime.utcfromtimestamp(dispatch_ts).isoformat(timespec="milliseconds") + "Z"

        with dispatch_stats_sampled_lock:
            dispatch_stats_sampled["lags"].append(dispatch_lag)

        logger.debug(f"[request #{request_id}] Dispatch confirmed at offset {dispatch_offset:.3f}s "
              f"(scheduled: {scheduled_offset:.3f}s, lag: {dispatch_lag:+.3f}s)")

        # Collect response(s) for this launcher
        outs = req_launcher.get_next_ready()

        # --- Release launcher immediately after responses are collected ---
        launcher_pool.put(req_launcher)
        launcher_released = True

        for out in outs:
            request_metrics, gen_text, _ = out

            response_mono = time.monotonic()
            response_offset = response_mono - start_time_mono

            response_ts = time.time()
            response_ts_utc = datetime.utcfromtimestamp(response_ts).isoformat(timespec="milliseconds") + "Z"

            logger.debug(f"[request #{request_id}] Response received at offset {response_offset:.3f}s")

            # Log to file
            with log_lock:
                log_fh.write(json.dumps({
                    "request_id": request_id,
                    "scheduled_offset_s": scheduled_offset,
                    "dispatch_offset_s": round(dispatch_offset, 3),
                    "dispatch_lag_s": round(dispatch_lag, 3),
                    "dispatch_ts_utc": dispatch_ts_utc,
                    "response_offset_s": round(response_offset, 3),
                    "response_ts_utc": response_ts_utc,
                }) + "\n")
                log_fh.flush()

            request_metrics = finalize_request_metrics(request_metrics, gen_text, common_metrics, get_token_length, request_id=request_id)
            with completed_sampled_lock:
                completed_sampled_requests.append(request_metrics)

    except Exception as e:
        logger.exception(f"[request #{request_id}] Exception during launch/record: {e}")

    finally:
        # --- Always return the launcher to the pool ---
        if not launcher_released:
            launcher_pool.put(req_launcher)


def _run_unsampled_records(
        sched: Dict[str, Any],
        start_time_mono: float,
        t0_utc: float,
        model: str,
        llm_api: str,
        additional_sampling_params: str,
        tokenizer: AutoTokenizer,
        get_token_length,
        log_lock: threading.Lock,
        log_fh,
        unsampled_counter: Dict[str, int],
        unsampled_lock: threading.Lock,
        dispatch_stats_unsampled: Dict[str, List[float]],
        dispatch_stats_unsampled_lock: threading.Lock,
):
    """
    Fire-and-forget version of scheduled request launcher.
    Does not stream, collect metrics, or use Ray — just dispatches a request
    to fully load the backend and logs the dispatch timestamp.
    """


    request_id = sched["request_id"]
    scheduled_offset = sched["scheduled_offset_s"]

    # --- Sleep until ~10s before scheduled time to prep request ---
    time.sleep(max(0, start_time_mono + scheduled_offset - time.monotonic() - 0.1))
    logger.debug(f"[request #{sched['request_id']}][unsampled] Preparing unsampled dispatch...")

    prep_start = time.monotonic()

    # --- Prepare request payload ---
    prompt = build_scheduled_sonnet_prompt(
        input_tokens=sched["input_tokens"],
        output_tokens=sched["output_tokens"],
        tokenizer=tokenizer
    )
    sampling_params = {"max_tokens": sched["output_tokens"]}
    sampling_params.update(json.loads(additional_sampling_params))


    logger.debug(f"[request #{request_id}][unsampled] Request prep completed with duration {time.monotonic() - prep_start:.3f}s")

    # --- Final sleep to align with schedule precisely ---
    time.sleep(max(0, start_time_mono + scheduled_offset - time.monotonic()))

    dispatch_ts_mono = time.monotonic()
    dispatch_offset = dispatch_ts_mono - start_time_mono
    dispatch_lag = dispatch_offset - scheduled_offset
    dispatch_ts = time.time()
    dispatch_ts_utc = datetime.utcfromtimestamp(dispatch_ts).isoformat(timespec="milliseconds") + "Z"

    with dispatch_stats_unsampled_lock:
        dispatch_stats_unsampled["lags"].append(dispatch_lag)

    logger.debug(
        f"[request #{request_id}][unsampled] Dispatching fire-and-forget at offset {dispatch_offset:.3f}s "
        f"(scheduled: {scheduled_offset:.3f}s, lag: {dispatch_lag:+.3f}s)"
    )


    # --- Dispatch logic (non-streaming, best-effort) ---
    try:
        if llm_api in ("litellm", "openai"):
            # Shared OpenAI-style request
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            }
            body.update(sampling_params or {})

            if llm_api == "litellm":
                from litellm import completion
                completion(**body)
            else:
                address = os.environ.get("OPENAI_API_BASE")
                key = os.environ.get("OPENAI_API_KEY")
                if not address or not key:
                    raise ValueError("Missing OPENAI_API_BASE or OPENAI_API_KEY.")
                if not address.endswith("/"):
                    address += "/"
                address += "chat/completions"
                headers = {"Authorization": f"Bearer {key}"}
                requests.post(address, json=body, headers=headers, timeout=60)

            with unsampled_lock:
                unsampled_counter["count"] += 1

        elif llm_api == "sagemaker":
            region = os.environ.get("AWS_REGION_NAME")
            if not region:
                raise ValueError("AWS_REGION_NAME must be set for SageMaker.")
            sm_runtime = boto3.client("sagemaker-runtime", region_name=region)
            message = {
                "inputs": [
                    [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": prompt},
                    ]
                ],
                "parameters": sampling_params,
            }
            sm_runtime.invoke_endpoint(
                EndpointName=model,
                ContentType="application/json",
                Body=json.dumps(message),
                CustomAttributes="accept_eula=true",
            )
            with unsampled_lock:
                unsampled_counter["count"] += 1

        elif llm_api == "vertexai":
            project_id = os.environ.get("GCLOUD_PROJECT_ID")
            region = os.environ.get("GCLOUD_REGION")
            endpoint_id = os.environ.get("VERTEXAI_ENDPOINT_ID")
            access_token = os.environ.get("GCLOUD_ACCESS_TOKEN", "").strip()
            if not all([project_id, region, endpoint_id, access_token]):
                raise ValueError("Missing required VertexAI env vars.")
            url = (
                f"https://{region}-aiplatform.googleapis.com/v1/projects/"
                f"{project_id}/locations/{region}/endpoints/{endpoint_id}:predict"
            )
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
            if "max_new_tokens" in sampling_params:
                sampling_params["maxOutputTokens"] = sampling_params.pop("max_new_tokens")
            data = {
                "instances": [{"prompt": prompt}],
                "parameters": sampling_params,
            }
            requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
            with unsampled_lock:
                unsampled_counter["count"] += 1

        else:
            logger.warning(f"[request #{request_id}][unsampled] Unsupported llm_api: {llm_api}")

    except Exception as e:
        logger.exception(f"[request #{request_id}][unsampled] Dispatch failed: {e}")

    # --- Log the dispatch event ---
    with log_lock:
        log_fh.write(json.dumps({
            "request_id": request_id,
            "scheduled_offset_s": scheduled_offset,
            "dispatch_offset_s": round(dispatch_offset, 3),
            "dispatch_lag_s": round(dispatch_lag, 3),
            "dispatch_ts_utc": dispatch_ts_utc,
            "unsampled": True,
        }) + "\n")
        log_fh.flush()




def get_token_throughput_latencies(
    model: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="openai",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    random.seed(11111)

    tokenizer = get_tokenizer_for_model(model)
    get_token_length = lambda text: len(tokenizer.encode(text))

    if not additional_sampling_params:
        additional_sampling_params = {}

    completed_requests_lock = threading.Lock()
    completed_requests = []
    num_completed_requests = 0
    # make up prompts outside of send loop for faster benchmarking loop
    num_output_tokens_list = []
    prompts = []
    for i in range(max_num_completed_requests):
        num_output_tokens = (sample_random_positive_int(
            mean_output_tokens, stddev_output_tokens
        ))
        num_output_tokens_list.append(num_output_tokens)

        prompts.append(randomly_sample_sonnet_lines_prompt(
            prompt_tokens_mean=mean_input_tokens,
            prompt_tokens_stddev=stddev_input_tokens,
            expect_output_tokens=num_output_tokens,
            tokenizer=tokenizer
        ))
    start_time = time.monotonic()
    pbar = tqdm(total=max_num_completed_requests)

    def launch_request(thread_index):
        nonlocal num_completed_requests
        clients = construct_clients(llm_api=llm_api, num_clients=1)
        req_launcher = RequestsLauncher(clients)
        request_index = thread_index % max_num_completed_requests

        while (
            time.monotonic() - start_time < test_timeout_s
            and num_completed_requests < max_num_completed_requests
        ):

            default_sampling_params = {"max_tokens": num_output_tokens_list[request_index] }
            default_sampling_params.update(additional_sampling_params)
            request_config = RequestConfig(
                model=model,
                prompt=prompts[request_index],
                sampling_params=default_sampling_params,
                llm_api=llm_api,
            )
            req_launcher.launch_requests(request_config)

            outs = req_launcher.get_next_ready()
            all_metrics = []
            for out in outs:
                request_metrics, gen_text, _ = out
                request_metrics = finalize_request_metrics(request_metrics, gen_text, common_metrics, get_token_length)
                with completed_requests_lock:
                    if num_completed_requests < max_num_completed_requests:
                        all_metrics.append(request_metrics)
                        completed_requests.extend(all_metrics)
                        pbar.update(len(all_metrics))
                        num_completed_requests += len(all_metrics)
                        request_index = (request_index + num_concurrent_requests) % max_num_completed_requests

    threads = []
    for i in range(num_concurrent_requests):
        thread = threading.Thread(target=launch_request, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    pbar.close()
    end_time = time.monotonic()
    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")

    # check one last time that there are no remaining results to collect.
    clients = construct_clients(llm_api=llm_api, num_clients=1)
    req_launcher = RequestsLauncher(clients)
    outs = req_launcher.get_next_ready()
    all_metrics = []
    for out in outs:
        request_metrics, gen_text, _ = out
        num_output_tokens = get_token_length(gen_text)
        with completed_requests_lock:
            if num_completed_requests < max_num_completed_requests:
                if num_output_tokens:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
                else:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
                request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
                request_metrics[common_metrics.NUM_TOTAL_TOKENS] = request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
                request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = num_output_tokens / request_metrics[common_metrics.E2E_LAT]
                completed_requests.extend(request_metrics)

    print(f"\Results for token benchmark for {model} queried with the {llm_api} api.\n")
    ret = metrics_summary(completed_requests, start_time, end_time)

    metadata = {
        "model": model,
        "mean_input_tokens": mean_input_tokens,
        "stddev_input_tokens": stddev_input_tokens,
        "mean_output_tokens": mean_output_tokens,
        "stddev_output_tokens": stddev_output_tokens,
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
    }

    metadata["results"] = ret

    return metadata, completed_requests

# Optional: import tiktoken if available
try:
    import tiktoken
except ImportError:
    tiktoken = None

_tokenizer_cache = {}

# Map of known model substrings → tokenizer loader function
KNOWN_TOKENIZERS = {
    # OpenAI models use tiktoken
    "gpt-4o": lambda: tiktoken.get_encoding("o200k_base") if tiktoken else None,
    "gpt-4-turbo": lambda: tiktoken.get_encoding("o200k_base") if tiktoken else None,
    "gpt-4": lambda: tiktoken.get_encoding("cl100k_base") if tiktoken else None,
    "gpt-3.5": lambda: tiktoken.get_encoding("cl100k_base") if tiktoken else None,
    "text-davinci": lambda: tiktoken.get_encoding("p50k_base") if tiktoken else None,
}

def get_tokenizer_for_model(model_name: str):
    """
    Attempts to load a tokenizer for a given model name.
    1. Tries known local mappings (e.g. OpenAI/tiktoken)
    2. Then tries Hugging Face AutoTokenizer
    3. Falls back to LlamaTokenizerFast
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    # 1. Known special cases (OpenAI, etc.)
    for key, fn in KNOWN_TOKENIZERS.items():
        if key in model_name.lower():
            tokenizer = fn()
            if tokenizer:
                logger.info(f"Using known tokenizer for model '{model_name}': {key}")
                logger.info(f"Tokenizer type: {type(tokenizer)}")
                _tokenizer_cache[model_name] = tokenizer
                return tokenizer
            else:
                logger.warning(
                    f"Known tokenizer for '{model_name}' requires `tiktoken` but it’s not installed."
                )

    # 2. Hugging Face fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        logger.info(f"Loaded HF tokenizer for model '{model_name}'.")
        _tokenizer_cache[model_name] = tokenizer
        return tokenizer
    except Exception as e:
        logger.warning(
            f"Failed to load tokenizer for '{model_name}' from Hugging Face ({e}). "
            "Falling back to LlamaTokenizerFast."
        )

    # 3. Llama fallback
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    _tokenizer_cache[model_name] = tokenizer
    logger.warning(f"USING THE FALLBACK 'LlamaTokenizerFast' TOKENIZER MAY RESULT IN INACCURATE TOKEN COUNTS WHEN USED WITH A MODEL WHICH USES A DIFFERENT TOKENIZER.")
    return tokenizer

def finalize_request_metrics(request_metrics, gen_text, common_metrics, get_token_length, request_id=None):
    """Normalize and augment request metrics with token and throughput info."""
    num_output_tokens = get_token_length(gen_text)

    ttft = request_metrics.get(common_metrics.TTFT, 0)
    itl_sum = request_metrics.get(common_metrics.INTER_TOKEN_LAT_SUM, 0)
    e2e = request_metrics.get(common_metrics.E2E_LAT, 0)

    expected_total = ttft + itl_sum

    # Warn if E2E differs significantly from TTFT + ITL_SUM
    if not math.isclose(expected_total, e2e, rel_tol=0.01, abs_tol=0.002):
        diff = e2e - expected_total
        pct_diff = (diff / e2e * 100) if e2e else float("nan")
        id_prefix = f"[request #{request_id}] " if request_id is not None else ""
        logger.warning(
            f"{id_prefix}"
            f"Latency mismatch: E2E ({e2e:.3f}s) vs TTFT+ITL_SUM ({expected_total:.3f}s) "
            f"Δ={diff:+.3f}s ({pct_diff:+.2f}%)"
        )

    # Calculate mean inter-token latency excluding first token
    if num_output_tokens > 1:
        request_metrics[common_metrics.INTER_TOKEN_LAT_MEAN] = (
                request_metrics[common_metrics.INTER_TOKEN_LAT_SUM] / (num_output_tokens - 1)
        )
    else:
        request_metrics[common_metrics.INTER_TOKEN_LAT_MEAN] = 0.0

    request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
    request_metrics[common_metrics.NUM_TOTAL_TOKENS] = (
            request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
    )
    request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
        num_output_tokens / request_metrics[common_metrics.E2E_LAT]
        if request_metrics[common_metrics.E2E_LAT]
        else 0
    )

    return request_metrics


def metrics_summary(
    metrics: List[Dict[str, Any]], start_time: int, end_time: int
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics: The metrics to summarize.
        start_time: The time the test started.
        end_time: The time the test ended.

    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Inter token latency
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret = {}

    def flatten(item):
        for sub_item in item:
            if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
                yield from flatten(sub_item)
            else:
                yield sub_item

    df = pd.DataFrame(metrics)
    df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]

    for key in [
        common_metrics.INTER_TOKEN_LAT_SUM,
        common_metrics.INTER_TOKEN_LAT_MEAN,
        common_metrics.TTFT,
        common_metrics.E2E_LAT,
        common_metrics.REQ_OUTPUT_THROUGHPUT,
        common_metrics.NUM_INPUT_TOKENS,
        common_metrics.NUM_OUTPUT_TOKENS
    ]:
        print(key)
        ret[key] = {}
        series = pd.Series(list(flatten(df_without_errored_req[key]))).dropna()
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        quantiles_reformatted_keys = {}
        for quantile, value in quantiles.items():
            reformatted_key = f"p{int(quantile * 100)}"
            print(f"    {reformatted_key} = {value}")
            quantiles_reformatted_keys[reformatted_key] = value
        ret[key]["quantiles"] = quantiles_reformatted_keys
        mean = series.mean()
        print(f"    mean = {mean}")
        ret[key]["mean"] = mean
        print(f"    min = {series.min()}")
        ret[key]["min"] = series.min()
        print(f"    max = {series.max()}")
        ret[key]["max"] = series.max()
        print(f"    stddev = {series.std()}")
        ret[key]["stddev"] = series.std()

    ret[common_metrics.NUM_REQ_STARTED] = len(metrics)

    error_codes = df[common_metrics.ERROR_CODE].dropna()
    num_errors = len(error_codes)
    ret[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
    ret[common_metrics.NUM_ERRORS] = num_errors
    print(f"Number Of Errored Requests: {num_errors}")
    error_code_frequency = dict(error_codes.value_counts())
    if num_errors:
        error_code_frequency = dict(error_codes.value_counts())
        print("Error Code Frequency")
        print(error_code_frequency)
    ret[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)

    overall_output_throughput = df_without_errored_req[
        common_metrics.NUM_OUTPUT_TOKENS
    ].sum() / (end_time - start_time)

    print(f"Overall Output Throughput: {overall_output_throughput}")
    ret[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

    num_completed_requests = len(df_without_errored_req)
    num_completed_requests_per_min = (
        num_completed_requests / (end_time - start_time) * 60
    )
    print(f"Number Of Completed Requests: {num_completed_requests}")
    print(f"Completed Requests Per Minute: {num_completed_requests_per_min}")

    ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min

    return ret


def run_token_benchmark(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: str,
    results_dir: str,
    user_metadata: Dict[str, Any],
    schedule_file: str = "", # optional; when set we route through the schedule branch
    schedule_file_subdir: bool = True,
    max_sampled_requests_per_second: int = 15,
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.
        user_metadata: Additional metadata to include in the results.
        schedule_file: Allows the use of a schedule file using monotonic offsets.
            First record should be given a time of 0.00. Format can be seen in schedule.csv.
            When this is set, stddevs are forced to 0.
    """
    if mean_input_tokens < 40:
        print(
            "the minimum number of input tokens that will be sent is 41"
            " because of the prompting logic right now"
        )

    # --- Branching point ---
    if schedule_file:
        print(f"[schedule mode] Using schedule runner. File: {schedule_file}")
        summary, individual_responses = run_schedule_mode(
            model=model,
            llm_api=llm_api,
            schedule_file=schedule_file,
            results_dir=results_dir,
            additional_sampling_params=additional_sampling_params,
            use_subdir=schedule_file_subdir,
            max_sampled_requests_per_second=max_sampled_requests_per_second
        )

        # Update to summary.
        summary.update(user_metadata)
        summary["schedule_file"] = schedule_file

        summary_filename = f"summary"
        results = LLMPerfResults(name=summary_filename, metadata=summary)

        # Override results_dir_path with the subdir actually used
        results_dir_path = Path(summary["results_subdir"])
        try:
            with open(results_dir_path / f"{summary_filename}.json", "w") as f:
                json.dump(results.to_dict(), f, indent=4, default=str)
        except Exception as e:
            print(results.to_dict())
            raise e

        try:
            with open(results_dir_path / "individual_responses.json", "w") as f:
                json.dump(individual_responses, f, indent=4)
        except Exception as e:
            print(individual_responses)
            raise e

    else:
        summary, individual_responses = get_token_throughput_latencies(
            model=model,
            llm_api=llm_api,
            test_timeout_s=test_timeout_s,
            max_num_completed_requests=max_num_completed_requests,
            mean_input_tokens=mean_input_tokens,
            stddev_input_tokens=stddev_input_tokens,
            mean_output_tokens=mean_output_tokens,
            stddev_output_tokens=stddev_output_tokens,
            num_concurrent_requests=num_concurrent_requests,
            additional_sampling_params=json.loads(additional_sampling_params),
        )
        if results_dir:
            filename = f"{model}_{mean_input_tokens}_{mean_output_tokens}"
            filename = re.sub(r"[^\w\d-]+", "-", filename)
            filename = re.sub(r"-{2,}", "-", filename)
            summary_filename = f"{filename}_summary"
            individual_responses_filename = f"{filename}_individual_responses"

            # Update to metadata.
            summary.update(user_metadata)

            results = LLMPerfResults(name=summary_filename, metadata=summary)
            results_dir = Path(results_dir)
            if not results_dir.exists():
                results_dir.mkdir(parents=True)
            elif not results_dir.is_dir():
                raise ValueError(f"{results_dir} is not a directory")

            try:
                with open(results_dir / f"{summary_filename}.json", "w") as f:
                    json.dump(results.to_dict(), f, indent=4, default=str)
            except Exception as e:
                print(results.to_dict())
                raise e

            try:
                with open(results_dir / f"{individual_responses_filename}.json", "w") as f:
                    json.dump(individual_responses, f, indent=4)
            except Exception as e:
                print(individual_responses)
                raise e


args = argparse.ArgumentParser(
    description="Run a token throughput and latency benchmark."
)

args.add_argument(
    "--model", type=str, required=True, help="The model to use for this load test."
)
args.add_argument(
    "--mean-input-tokens",
    type=int,
    default=550,
    help=(
        "The mean number of tokens to send in the prompt for the request. "
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-input-tokens",
    type=int,
    default=150,
    help=(
        "The standard deviation of number of tokens to send in the prompt for the request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--mean-output-tokens",
    type=int,
    default=150,
    help=(
        "The mean number of tokens to generate from each llm request. This is the max_tokens param "
        "for the completions API. Note that this is not always the number of tokens returned. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-output-tokens",
    type=int,
    default=80,
    help=(
        "The stdandard deviation on the number of tokens to generate per llm request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--num-concurrent-requests",
    type=int,
    default=10,
    help=("The number of concurrent requests to send (default: %(default)s)"),
)
args.add_argument(
    "--timeout",
    type=int,
    default=90,
    help="The amount of time to run the load test for. (default: %(default)s)",
)
args.add_argument(
    "--max-num-completed-requests",
    type=int,
    default=10,
    help=(
        "The number of requests to complete before finishing the test. Note "
        "that its possible for the test to timeout first. (default: %(default)s)"
    ),
)
args.add_argument(
    "--additional-sampling-params",
    type=str,
    default="{}",
    help=(
        "Additional sampling params to send with the each request to the LLM API. "
        "(default: %(default)s) No additional sampling params are sent."
    ),
)
args.add_argument(
    "--results-dir",
    type=str,
    default="",
    help=(
        "The directory to save the results to. "
        "(`default: %(default)s`) No results are saved)"
    ),
)
args.add_argument(
    "--llm-api",
    type=str,
    default="openai",
    help=(
        f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--metadata",
    type=str,
    default="",
    help=(
        "A comma separated list of metadata to include in the results, e.g. "
        "name=foo,bar=1. These will be added to the metadata field of the results. "
    ),
)

args.add_argument(
    "--schedule-file-subdir",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Whether to place scheduled output files in a subdirectory named after the schedule file "
        "(default: True). Use --no-schedule-file-subdir to disable."
    ),
)

args.add_argument(
    "--schedule-file",
    type=str,
    default="",
    help=(
        "Optional CSV with rows: delta_from_t0,input_tokens,output_tokens. "
        "When provided, stddevs are forced to 0."
    ),
)

args.add_argument(
    "--max-sampled-requests-per-second",
    type=int,
    default=15,
    help=(
        "When using --schedule-file, this caps the rate of sampled requests per second. "
        "(default: %(default)s)"
        "All unsampled requests are sent as fast as possible without inference metrics collected."
    ),
)

args.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging verbosity level (default: INFO).",
)

def setup_logger(level: str):
    """Thread-safe logger with adjustable verbosity."""
    logger = logging.getLogger("benchmark")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear old handlers if re-run interactively
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(handler)

    # Silence noisy deps
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    return logger

if __name__ == "__main__":
    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})
    args = args.parse_args()


    # Set up logger before anything noisy
    logger = setup_logger(args.log_level)
    logger.info(f"Log level set to {args.log_level}")
    logging.Formatter.converter = time.gmtime


    # Parse user metadata.
    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    if args.schedule_file:
        IGNORED_ARGS_IN_SCHEDULE_MODE = {
            "--stddev-input-tokens": args.stddev_input_tokens != 150,
            "--stddev-output-tokens": args.stddev_output_tokens != 80,
            "--num-concurrent-requests": args.num_concurrent_requests != 10,
            "--timeout": args.timeout != 90,
            "--mean-input-tokens": args.mean_input_tokens != 550,
            "--mean-output-tokens": args.mean_output_tokens != 150,
        }
        ignored = [arg for arg, was_set in IGNORED_ARGS_IN_SCHEDULE_MODE.items() if was_set]
        if ignored:
            print(
                f"⚠️  Warning: The following arguments will be ignored due to --schedule-file mode:\n"
                f"   {', '.join(ignored)}"
            )

    run_token_benchmark(
        llm_api=args.llm_api,
        model=args.model,
        test_timeout_s=args.timeout,
        max_num_completed_requests=args.max_num_completed_requests,
        mean_input_tokens=args.mean_input_tokens,
        stddev_input_tokens=args.stddev_input_tokens,
        mean_output_tokens=args.mean_output_tokens,
        stddev_output_tokens=args.stddev_output_tokens,
        num_concurrent_requests=args.num_concurrent_requests,
        additional_sampling_params=args.additional_sampling_params,
        results_dir=args.results_dir,
        user_metadata=user_metadata,
        schedule_file=args.schedule_file,
        schedule_file_subdir=args.schedule_file_subdir,
        max_sampled_requests_per_second=args.max_sampled_requests_per_second
    )
