import threading
import argparse
from collections.abc import Iterable
import json
import os
from pathlib import Path
import re
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import ray

from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients

from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    randomly_sample_sonnet_lines_prompt,
    LLMPerfResults,
    sample_random_positive_int,
)
from tqdm import tqdm

from transformers import LlamaTokenizerFast

def run_schedule_mode(
        *,
        llm_api: str,
        model: str,
        schedule_file: str,
        results_dir: str,
        additional_sampling_params: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Dispatch requests using a schedule file with delta timestamps.
    Preserves concurrency by launching requests in background threads at their appropriate offsets.
    """
    import csv
    from datetime import datetime
    from concurrent.futures import ThreadPoolExecutor

    from shutil import copyfile

    # Make timestamped subdir for this run
    utc_time = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    subdir_name = f"{utc_time}_schedule_run"
    results_subdir_path = Path(results_dir) / subdir_name
    results_subdir_path.mkdir(parents=True, exist_ok=True)

    # Copy schedule file into subdir for recordkeeping
    copyfile(schedule_file, results_subdir_path / "schedule.csv")

    # Log file inside subdir
    log_path = results_subdir_path / "requests_sent.log"

    # Prepare tokenizer
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    get_token_length = lambda text: len(tokenizer.encode(text))

    # Read schedule and attach request IDs
    with open(schedule_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        schedule = []
        for idx, row in enumerate(reader):
            schedule.append({
                "request_id": idx + 1,
                "scheduled_offset_s": float(row["scheduled_offset_s"]),
                "input_tokens": int(row["input_tokens"]),
                "output_tokens": int(row["output_tokens"]),
            })

    base_time = time.monotonic()
    t0_utc = time.time()
    log_fh = open(log_path, "a")

    # Shared results
    completed_requests = []
    completed_lock = threading.Lock()

    def launch_and_record(sched: Dict[str, Any]):
        request_id = sched["request_id"]
        scheduled_offset = sched["scheduled_offset_s"]

        # Prepare everything we can *before* waiting
        prompt = randomly_sample_sonnet_lines_prompt(
            prompt_tokens_mean=sched["input_tokens"],
            prompt_tokens_stddev=0,
            expect_output_tokens=sched["output_tokens"],
            tokenizer=tokenizer
        )

        default_sampling_params = {"max_tokens": sched["output_tokens"]}
        default_sampling_params.update(json.loads(additional_sampling_params))

        # Sleep until scheduled time
        time.sleep(max(0, base_time + scheduled_offset - time.monotonic()))

        dispatch_ts_mono = time.monotonic()
        dispatch_offset = dispatch_ts_mono - base_time
        dispatch_lag = dispatch_offset - scheduled_offset

        dispatch_ts = time.time()
        dispatch_ts_utc = datetime.utcfromtimestamp(dispatch_ts).isoformat(timespec="milliseconds") + "Z"

        print(f"[request #{request_id}] Dispatching at offset {dispatch_offset:.3f}s "
              f"(scheduled: {scheduled_offset:.3f}s, lag: {dispatch_lag:+.3f}s)")

        request_config = RequestConfig(
            model=model,
            prompt=prompt,
            sampling_params=default_sampling_params,
            llm_api=llm_api,
        )

        clients = construct_clients(llm_api=llm_api, num_clients=1)
        req_launcher = RequestsLauncher(clients)
        req_launcher.launch_requests(request_config)

        outs = req_launcher.get_next_ready()
        for out in outs:
            request_metrics, gen_text, _ = out
            response_ts = time.time()
            response_offset = response_ts - t0_utc
            response_ts_utc = datetime.utcfromtimestamp(response_ts).isoformat(timespec="milliseconds") + "Z"

            print(f"[request #{request_id}] Response received at offset {response_offset:.3f}s")

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

            num_output_tokens = get_token_length(gen_text)
            if num_output_tokens:
                request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
            else:
                request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
            request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
            request_metrics[common_metrics.NUM_TOTAL_TOKENS] = request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
            request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = num_output_tokens / request_metrics[common_metrics.E2E_LAT]

            with completed_lock:
                completed_requests.append(request_metrics)


    # Run requests on threads with natural concurrency
    with ThreadPoolExecutor(max_workers=500) as executor:
        for sched in schedule:
            executor.submit(launch_and_record, sched)

    executor.shutdown(wait=True)
    log_fh.close()

    print(f"\nResults for schedule-mode benchmark for {model} queried with {llm_api} API.\n")

    start_time = base_time
    end_time = time.monotonic()
    summary = metrics_summary(completed_requests, start_time, end_time)

    summary.update({
        "model": model,
        "num_concurrent_requests": "scheduled",  # Not fixed concurrency
        "num_launched": len(schedule),
        "schedule_file": schedule_file,
        "wall_time_s": end_time - start_time,
    })
    summary["results_subdir"] = str(results_subdir_path)

    return summary, completed_requests


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

    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )
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
                num_output_tokens = get_token_length(gen_text)
                with completed_requests_lock:
                    if num_completed_requests < max_num_completed_requests:
                        if num_output_tokens:
                            request_metrics[common_metrics.INTER_TOKEN_LAT] /= request_metrics[common_metrics.NUM_OUTPUT_TOKENS]
                        else:
                            request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
                        request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
                        request_metrics[common_metrics.NUM_TOTAL_TOKENS] = request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
                        request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = num_output_tokens / request_metrics[common_metrics.E2E_LAT]
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
        common_metrics.INTER_TOKEN_LAT,
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
        schedule_file: str = "",  # optional; when set we route through the schedule branch
):
    """
    Run either baseline or schedule mode benchmark.
    """
    if mean_input_tokens < 40:
        print(
            "The minimum number of input tokens that will be sent is 41 "
            "because of the prompting logic right now."
        )

    # Default results dir
    results_dir_path = Path(results_dir)

    # --- Branching point ---
    if schedule_file:
        print(f"[schedule mode] Using schedule runner. File: {schedule_file}")
        summary, individual_responses = run_schedule_mode(
            llm_api=llm_api,
            model=model,
            schedule_file=schedule_file,
            results_dir=results_dir,
            additional_sampling_params=additional_sampling_params,
        )
        # Override results_dir_path with the subdir actually used
        results_dir_path = Path(summary["results_subdir"])
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

        summary.update(user_metadata)
        if schedule_file:
            summary["schedule_file"] = schedule_file

        results = LLMPerfResults(name=summary_filename, metadata=summary)

        try:
            with open(results_dir_path / f"{summary_filename}.json", "w") as f:
                json.dump(results.to_dict(), f, indent=4, default=str)
        except Exception as e:
            print(results.to_dict())
            raise e

        try:
            with open(results_dir_path / f"{individual_responses_filename}.json", "w") as f:
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

# NEW: optional replay/scheduler file (just parsed here; behavior wired later)
args.add_argument(
    "--schedule-file",
    type=str,
    default="",
    help=(
        "Optional CSV with rows: delta_from_t0,input_tokens,output_tokens. "
        "When provided, stddevs are forced to 0."
    ),
)


if __name__ == "__main__":
    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})
    args = argparse.ArgumentParser(
        description="Run a token throughput and latency benchmark."
    )

    args.add_argument("--model", type=str, required=True)
    args.add_argument("--mean-input-tokens", type=int, default=550)
    args.add_argument("--stddev-input-tokens", type=int, default=150)
    args.add_argument("--mean-output-tokens", type=int, default=150)
    args.add_argument("--stddev-output-tokens", type=int, default=80)
    args.add_argument("--num-concurrent-requests", type=int, default=10)
    args.add_argument("--timeout", type=int, default=90)
    args.add_argument("--max-num-completed-requests", type=int, default=10)
    args.add_argument("--additional-sampling-params", type=str, default="{}")
    args.add_argument("--results-dir", type=str, default="")
    args.add_argument("--llm-api", type=str, default="openai")
    args.add_argument("--metadata", type=str, default="")
    args.add_argument(
        "--schedule-file",
        type=str,
        default="",
        help="Optional CSV with rows: delta_from_t0,input_tokens,output_tokens. Enables schedule mode.",
    )

    parsed = args.parse_args()

    # Metadata
    user_metadata = {}
    if parsed.metadata:
        for item in parsed.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    # Force stddevs to 0 when using --schedule-file
    if parsed.schedule_file:
        parsed.stddev_input_tokens = 0
        parsed.stddev_output_tokens = 0

    run_token_benchmark(
        llm_api=parsed.llm_api,
        model=parsed.model,
        test_timeout_s=parsed.timeout,
        max_num_completed_requests=parsed.max_num_completed_requests,
        mean_input_tokens=parsed.mean_input_tokens,
        stddev_input_tokens=parsed.stddev_input_tokens,
        mean_output_tokens=parsed.mean_output_tokens,
        stddev_output_tokens=parsed.stddev_output_tokens,
        num_concurrent_requests=parsed.num_concurrent_requests,
        additional_sampling_params=parsed.additional_sampling_params,
        results_dir=parsed.results_dir,
        user_metadata=user_metadata,
        schedule_file=parsed.schedule_file,
    )
