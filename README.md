# LLMPerf

A Tool for evaulation the performance of LLM APIs.

# Installation
```bash
git clone https://github.com/ray-project/llmperf.git
cd llmperf
pip install -e .
```

# Basic Usage

We implement 2 tests for evaluating LLMs: a load test to check for performance and a correctness test to check for correctness.

## Load test

Load testing runs a number of requets to the LLM API and measures the inter-token latency and generation throughput per request and across measured requests.

Load testing can be run in two modes: a basic mode where the user specifies the number of concurrent requests and the token counts for the prompts and generations, and a schedule mode where the user provides a CSV file with a custom request schedule.


### Basic Mode
In order to run the most basic load test you can run the token_benchmark_ray script without providing a schedule file.

The load test spawns a number of concurrent requests to the LLM API and measures the inter-token latency and generation throughput per request and across concurrent requests. The prompt that is sent with each request is of the format:

```
Randomly stream lines from the following text. Don't generate eos tokens:
LINE 1,
LINE 2,
LINE 3,
...
```

Where the lines are randomly sampled from a collection of lines from Shakespeare sonnets. Tokens are counted using the `LlamaTokenizer` regardless of which LLM API is being tested. This is to ensure that the prompts are consistent across different LLM APIs.

To run the most basic load test you can the token_benchmark_ray script.

---

### Schedule Mode

You can optionally run the benchmark using a **CSV schedule file**. 
Running in schedule mode runs load test with a custom request schedule f, allowing you to reproducibly model bursty or irregular traffic patterns.
This file specifies exact dispatch times in an offset from `t0.00`, and token counts for each request. 
When this mode is enabled via `--schedule-file`, other token and concurrency-related arguments are **ignored**.

Some schedule files may submit requests at a rate which is higher than the framework can measure streaming requests. 
In these cases only some requests will be measured; Other requests will be dispatched with no timing information recorded.

#### Schedule file format

The CSV should include:

```
scheduled_offset_s,input_tokens,output_tokens
0.000,600,200
0.050,550,150
0.100,500,100
...
```

Each row corresponds to a single request:
- `scheduled_offset_s`: how many seconds after t=0 to launch the request
- `input_tokens`: number of tokens in the prompt
- `output_tokens`: number of tokens to generate

#### Schedule File Measurement Samples

When using a schedule file, `--max-sampled-requests-per-second` (default: `15/s`) is used to control the maximum number of measured requests which will be sent per second. 
All other requests will be sent, but not measured. Non-measured requests will be sent without streaming, and will not have timing information recorded. 



#### Ignored Parameters in Schedule Mode

When you provide `--schedule-file`, several other arguments become irrelevant, since the test configuration is fully driven by the schedule file.

The following arguments are **ignored** when `--schedule-file` is set:

- `--mean-input-tokens`
- `--stddev-input-tokens`
- `--mean-output-tokens`
- `--stddev-output-tokens`
- `--num-concurrent-requests`
- `--timeout`
- `--max-num-completed-requests`

If any of these arguments are set to non-default values while `--schedule-file` is used, the script will emit a warning to clarify they are being ignored.

---

#### Example Call with Schedule File

``​`bash
python token_benchmark_ray.py \
  --model "meta-llama/Llama-2-7b-chat-hf" \
  --schedule-file "schedules/bursty_schedule.csv" \
  --results-dir "result_outputs" \
  --llm-api "openai" \
  --additional-sampling-params '{}'
``​`

This will:
- Launch requests according to the timestamps in `bursty_schedule.csv`
- Save logs and results in a timestamped subdirectory of `result_outputs`
- Record per-request dispatch and response timing in `requests_sent.log`
- Copy the schedule file into the results directory for traceability

#### Notes

- Requests are dispatched in separate threads to maintain natural concurrency.
- Prompt templates and token counting still use the Llama tokenizer.
- This mode is useful for replaying or modeling bursty traffic patterns.

#### Output

When using `--schedule-file`, results are saved in a **timestamped subdirectory** inside `--results-dir`. That subdirectory contains:
- `summary.json`: performance metrics summary
- `individual_responses.json`: per-request metrics
- `requests_sent.log`: exact dispatch/response times and lag
- `schedule.csv`: copy of the schedule file used

You will see a warning if you supply arguments that will be ignored in schedule mode (e.g. `--stddev-input-tokens`, `--timeout`, etc.).


### Load Test Results


#### Files
The results of the load test are saved in the results directory specified by the `--results-dir` argument.

The results always save two files, one with the summary metrics of the test, and one with metrics from each individual request that is returned: 

`<model>_<isl>_<osl>_individual_responses.json`

`<model>_<isl>_<osl>_summary.json`

Where `<model>` is the model name, `<isl>` is the input sequence length, and `<osl>` is the output sequence length.

Additionally, if you run in schedule mode, the schedule file is copied to the results directory for traceability and a log of all requests that were sent with their dispatch and response times is saved:

`requests_sent.log`

`schedule.csv`

#### Metrics

The `individual_responses.json` file contains a collection of json objects with the following metrics:

```
{
    "error_code": null, # null if no error
    "error_msg": "", # empty if no error
    "sum_inter_token_latency_s": 1.762017541041132, # sum of inter-token latencies in seconds. Excludes the latency for the first token.
    "mean_inter_token_latency_s": 0.017798156980213455, # mean inter-token latency in seconds. Sum of inter-token latencies divided by number of output tokens - 1.
    "ttft_s": 0.9066180440131575, # time to first token in seconds. Time from sending the request to receiving the first token.
    "end_to_end_latency_s": 2.6687197270221077, # end to end latency in seconds. Time from sending the request to receiving the last token.
    "start_time": 1760448727.766322, # start time of the request in epoch seconds.
    "end_time": 1760448730.446413, # end time of the request in epoch seconds.
    "request_output_throughput_token_per_s": 37.47115104949033, # output throughput in tokens per second. Number of output tokens divided by end to end latency.
    "number_total_tokens": 500, # total number of tokens (input + output)
    "number_output_tokens": 100, # number of output tokens (calculated by the tokenizer llmperf loads at runtime)
    "number_input_tokens": 400 # number of input tokens (calculated by the tokenizer llmperf loads at runtime)
}
```

The `summary.json` file contains a single json object with the following metrics:

```
{
    "version": "2025-10-14", # version of this fork of llmperf
    "name": "summary", # name of the metrics
    "sum_inter_token_latency_s_quantiles_p25": 1.447126763072447, # 25th percentile of sum of inter-token latencies in seconds
    "sum_inter_token_latency_s_quantiles_p50": 2.041566784173483, # 50th percentile of sum of inter-token latencies in seconds
    "sum_inter_token_latency_s_quantiles_p75": 2.129703682498075, # 75th percentile of sum of inter-token latencies in seconds
    "sum_inter_token_latency_s_quantiles_p90": 2.5180071656533984, # 90th percentile of sum of inter-token latencies in seconds
    "sum_inter_token_latency_s_quantiles_p95": 2.7555328389222264, # 95th percentile of sum of inter-token latencies in seconds
    "sum_inter_token_latency_s_quantiles_p99": 2.9455533775372893, # 99th percentile of sum of inter-token latencies in seconds
    "sum_inter_token_latency_s_mean": 1.8929514053577026, # mean of sum of inter-token latencies in seconds
    "sum_inter_token_latency_s_min": 1.0623736499983352, # minimum of sum of inter-token latencies in seconds
    "sum_inter_token_latency_s_max": 2.9930585121910553, # maximum of sum of inter-token latencies in seconds
    "sum_inter_token_latency_s_stddev": 0.6633714839631771, # standard deviation of sum of inter-token latencies in seconds
    "mean_inter_token_latency_s_quantiles_p25": 0.014617442051236837, # 25th percentile of mean inter-token latencies in seconds
    "mean_inter_token_latency_s_quantiles_p50": 0.020621886708823062, # 50th percentile of mean inter-token latencies in seconds
    "mean_inter_token_latency_s_quantiles_p75": 0.021512158409071468, # 75th percentile of mean inter-token latencies in seconds
    "mean_inter_token_latency_s_quantiles_p90": 0.025434415814680793, # 90th percentile of mean inter-token latencies in seconds
    "mean_inter_token_latency_s_quantiles_p95": 0.027833665039618448, # 95th percentile of mean inter-token latencies in seconds
    "mean_inter_token_latency_s_quantiles_p99": 0.02975306441956858, # 99th percentile of mean inter-token latencies in seconds
    "mean_inter_token_latency_s_mean": 0.01912072126623942, # mean of mean inter-token latencies in seconds
    "mean_inter_token_latency_s_min": 0.010731046969680153, # minimum of mean inter-token latencies in seconds
    "mean_inter_token_latency_s_max": 0.030232914264556116, # maximum of mean inter-token latencies in seconds
    "mean_inter_token_latency_s_stddev": 0.006700722060234113, # standard deviation of mean inter-token latencies in seconds
    "ttft_s_quantiles_p25": 0.6707058555039112, # 25th percentile of time to first token in seconds
    "ttft_s_quantiles_p50": 0.7314752560050692, # 50th percentile of time to first token in seconds
    "ttft_s_quantiles_p75": 0.8252051705057966, # 75th percentile of time to first token in seconds 
    "ttft_s_quantiles_p90": 0.9744803208042868, # 90th percentile of time to first token in seconds
    "ttft_s_quantiles_p95": 1.0253770283976336, # 95th percentile of time to first token in seconds 
    "ttft_s_quantiles_p99": 1.0660943944723111, # 99th percentile of time to first token in seconds
    "ttft_s_mean": 0.750521554428685, # mean of time to first token in seconds
    "ttft_s_min": 0.45407983698532917, # minimum of time to first token in seconds
    "ttft_s_max": 1.0762737359909806, # maximum of time to first token in seconds
    "ttft_s_stddev": 0.1965236596108972, # standard deviation of time to first token in seconds 
    "end_to_end_latency_s_quantiles_p25": 2.432024793990422, # 25th percentile of end to end latency in seconds
    "end_to_end_latency_s_quantiles_p50": 2.6687197270221077, # 50th percentile of end to end latency in seconds
    "end_to_end_latency_s_quantiles_p75": 2.787556847499218, # 75th percentile of end to end latency in seconds
    "end_to_end_latency_s_quantiles_p90": 3.150948110193713, # 90th percentile of end to end latency in seconds
    "end_to_end_latency_s_quantiles_p95": 3.412665084097534, # 95th percentile of end to end latency in seconds
    "end_to_end_latency_s_quantiles_p99": 3.622038663220591, # 99th percentile of end to end latency in seconds
    "end_to_end_latency_s_mean": 2.643558538290173, # mean of end to end latency in seconds
    "end_to_end_latency_s_min": 1.722644700028468, # minimum of end to end latency in seconds
    "end_to_end_latency_s_max": 3.674382058001356, # maximum of end to end latency in seconds
    "end_to_end_latency_s_stddev": 0.597571434909786, # standard deviation of end to end latency in seconds 
    "request_output_throughput_token_per_s_quantiles_p25": 35.87466990080029, # 25th percentile of output throughput in tokens per second
    "request_output_throughput_token_per_s_quantiles_p50": 37.47115104949033, # 50th percentile of output throughput in tokens per second
    "request_output_throughput_token_per_s_quantiles_p75": 41.46804645924712, # 75th percentile of output throughput in tokens per second
    "request_output_throughput_token_per_s_quantiles_p90": 50.386904016086774, # 90th percentile of output throughput in tokens per second
    "request_output_throughput_token_per_s_quantiles_p95": 54.2185899248603, # 95th percentile of output throughput in tokens per second
    "request_output_throughput_token_per_s_quantiles_p99": 57.28393865187913, # 99th percentile of output throughput in tokens per second
    "request_output_throughput_token_per_s_mean": 39.63176002402042, # mean of output throughput in tokens per second
    "request_output_throughput_token_per_s_min": 27.215460564923944, # minimum of output throughput in tokens per second
    "request_output_throughput_token_per_s_max": 58.05027583363385, # maximum of output throughput in tokens per second
    "request_output_throughput_token_per_s_stddev": 9.679446895441021, # standard deviation of output throughput in tokens per second 
    "number_input_tokens_quantiles_p25": 400.0, # 25th percentile of number of input tokens 
    "number_input_tokens_quantiles_p50": 400.0, # 50th percentile of number of input tokens
    "number_input_tokens_quantiles_p75": 400.0, # 75th percentile of number of input tokens
    "number_input_tokens_quantiles_p90": 400.0, # 90th percentile of number of input tokens
    "number_input_tokens_quantiles_p95": 400.0, # 95th percentile of number of input tokens
    "number_input_tokens_quantiles_p99": 400.0, # 99th percentile of number of input tokens
    "number_input_tokens_mean": 400.0, # mean of number of input tokens
    "number_input_tokens_min": "400", # minimum of number of input tokens
    "number_input_tokens_max": "400", # maximum of number of input tokens
    "number_input_tokens_stddev": 0.0, # standard deviation of number of input tokens
    "number_output_tokens_quantiles_p25": 100.0, # 25th percentile of number of output tokens
    "number_output_tokens_quantiles_p50": 100.0, # 50th percentile of number of output tokens
    "number_output_tokens_quantiles_p75": 100.0, # 75th percentile of number of output tokens
    "number_output_tokens_quantiles_p90": 100.0, # 90th percentile of number of output tokens 
    "number_output_tokens_quantiles_p95": 100.0, # 95th percentile of number of output tokens
    "number_output_tokens_quantiles_p99": 100.0, # 99th percentile of number of output tokens
    "number_output_tokens_mean": 100.0, # mean of number of output tokens
    "number_output_tokens_min": "100", # minimum of number of output tokens
    "number_output_tokens_max": "100", # maximum of number of output tokens
    "number_output_tokens_stddev": 0.0, # standard deviation of number of output tokens
    "num_requests_started": 7, # number of requests that were started 
    "error_rate": 0.0, # error rate (number of errors / number of requests started)
    "number_errors": 0, # number of errors
    "error_code_frequency": "{}", # frequency of error codes 
    "mean_output_throughput_token_per_s": 31.78765981870924, # mean output throughput across all requests in tokens per second (total output tokens / total end to end latency)
    "num_completed_requests": 7, # number of requests that were completed
    "num_completed_requests_per_min": 19.072595891225543, # number of completed requests per minute
    "model": "gpt-4o", # model name
    "num_concurrent_requests": "scheduled", # number of concurrent requests (or "scheduled" if using a schedule file)
    "num_launched": 7, # number of requests that were launched
    "num_unsampled": 0, # number of requests that were launched but not sampled (only when using a schedule file)
    "schedule_file": "schedule_files/schedule.csv", # path to the schedule file (only when using a schedule file)
    "results_subdir": "result_outputs/2025-10-14T13-31-52_schedule_run", # path to the results subdirectory
    "wall_time_s": 22.039070113009075, # wall time of the entire test in seconds
    "timestamp": 1760448749 # timestamp of the test in epoch seconds
}
```

#### Notes on Metrics

**Time to first token (TTFT)**: The time from sending the request to receiving the first token. This time is not included in the inter-token latency measurements.

**Inter Token Latency (ITL)**: The time between receiving each token. The sum of inter-token latencies is the sum of the time between receiving each token, excluding the time to first token. The mean inter-token latency is the sum of inter-token latencies divided by the number of output tokens minus one (since there are n-1 intervals between n tokens).
``` SUM_ITL = sum(ITL) ```
``` MEAN_ITL = SUM_ITL / (N_output_tokens - 1) ```

**Number of Tokens**: The number of tokens is calculated using the tokenizer that best fits the model being tested if known, otherwise the LlamaTokenizerFast is used.

**End to End Latency (E2E)**: The time from sending the request to receiving the last token. This time includes the time to first token and the sum of inter-token latencies.

**E2E == SUM_ITL + TTFT**: In the case that the sum of inter token latency plus time to first token _does NOT equal_ the end to end latency within a tolerance of 1% of (FFT + SUM_ITL)  OR .002s (whichever is larger) a warning is logged.



---


### Caveats and Disclaimers

- The endpoints provider backend might vary widely, so this is not a reflection on how the software runs on a particular hardware.
- The results may vary with time of day.
- The results may vary with the load.
- The results may not correlate with users’ workloads.

### OpenAI Compatible APIs
```bash
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE="https://api.endpoints.anyscale.com/v1"

python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api openai \
--additional-sampling-params '{}'

```

### Anthropic
```bash
export ANTHROPIC_API_KEY=secret_abcdefg

python token_benchmark_ray.py \
--model "claude-2" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api anthropic \
--additional-sampling-params '{}'

```

### TogetherAI

```bash
export TOGETHERAI_API_KEY="YOUR_TOGETHER_KEY"

python token_benchmark_ray.py \
--model "together_ai/togethercomputer/CodeLlama-7b-Instruct" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### Hugging Face

```bash
export HUGGINGFACE_API_KEY="YOUR_HUGGINGFACE_API_KEY"
export HUGGINGFACE_API_BASE="YOUR_HUGGINGFACE_API_ENDPOINT"

python token_benchmark_ray.py \
--model "huggingface/meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### LiteLLM

LLMPerf can use LiteLLM to send prompts to LLM APIs. To see the environment variables to set for the provider and arguments that one should set for model and additional-sampling-params.

see the [LiteLLM Provider Documentation](https://docs.litellm.ai/docs/providers).

```bash
python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### Vertex AI

Here, --model is used for logging, not for selecting the model. The model is specified in the Vertex AI Endpoint ID.

The GCLOUD_ACCESS_TOKEN needs to be somewhat regularly set, as the token generated by `gcloud auth print-access-token` expires after 15 minutes or so.

Vertex AI doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.

```bash

gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
export GCLOUD_REGION=YOUR_REGION
export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "vertexai" \
--additional-sampling-params '{}'

```

### SageMaker

SageMaker doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.

```bash

export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"s
export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
export AWS_REGION_NAME="YOUR_ENDPOINTS_REGION_NAME"

python llm_correctness.py \
--model "llama-2-7b" \
--llm-api "sagemaker" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

see `python token_benchmark_ray.py --help` for more details on the arguments.

## Correctness Test

The correctness test spawns a number of concurrent requests to the LLM API with the following format:

```
Convert the following sequence of words into a number: {random_number_in_word_format}. Output just your final answer.
```

where random_number_in_word_format could be for example "one hundred and twenty three". The test then checks that the response contains that number in digit format which in this case would be 123.

The test does this for a number of randomly generated numbers and reports the number of responses that contain a mismatch.

To run the most basic correctness test you can run the the llm_correctness.py script.

### OpenAI Compatible APIs

```bash
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE=https://console.endpoints.anyscale.com/m/v1

python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--max-num-completed-requests 150 \
--timeout 600 \
--num-concurrent-requests 10 \
--results-dir "result_outputs"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=secret_abcdefg

python llm_correctness.py \
--model "claude-2" \
--llm-api "anthropic"  \
--max-num-completed-requests 5 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs"
```

### TogetherAI

```bash
export TOGETHERAI_API_KEY="YOUR_TOGETHER_KEY"

python llm_correctness.py \
--model "together_ai/togethercomputer/CodeLlama-7b-Instruct" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### Hugging Face

```bash
export HUGGINGFACE_API_KEY="YOUR_HUGGINGFACE_API_KEY"
export HUGGINGFACE_API_BASE="YOUR_HUGGINGFACE_API_ENDPOINT"

python llm_correctness.py \
--model "huggingface/meta-llama/Llama-2-7b-chat-hf" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### LiteLLM

LLMPerf can use LiteLLM to send prompts to LLM APIs. To see the environment variables to set for the provider and arguments that one should set for model and additional-sampling-params.

see the [LiteLLM Provider Documentation](https://docs.litellm.ai/docs/providers).

```bash
python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

see `python llm_correctness.py --help` for more details on the arguments.


### Vertex AI

Here, --model is used for logging, not for selecting the model. The model is specified in the Vertex AI Endpoint ID.

The GCLOUD_ACCESS_TOKEN needs to be somewhat regularly set, as the token generated by `gcloud auth print-access-token` expires after 15 minutes or so.

Vertex AI doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.


```bash

gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
export GCLOUD_REGION=YOUR_REGION
export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--llm-api "vertexai" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### SageMaker

SageMaker doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.

```bash

export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"s
export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
export AWS_REGION_NAME="YOUR_ENDPOINTS_REGION_NAME"

python llm_correctness.py \
--model "llama-2-7b" \
--llm-api "sagemaker" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

## Saving Results

The results of the load test and correctness test are saved in the results directory specified by the `--results-dir` argument. The results are saved in 2 files, one with the summary metrics of the test, and one with metrics from each individual request that is returned.

# Advanced Usage

The correctness tests were implemented with the following workflow in mind:

```python
import ray
from transformers import LlamaTokenizerFast

from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher


# Copying the environment variables and passing them to ray.init() is necessary
# For making any clients work.
ray.init(runtime_env={"env_vars": {"OPENAI_API_BASE" : "https://api.endpoints.anyscale.com/v1",
                                   "OPENAI_API_KEY" : "YOUR_API_KEY"}})

base_prompt = "hello_world"
tokenizer = LlamaTokenizerFast.from_pretrained(
    "hf-internal-testing/llama-tokenizer"
)
base_prompt_len = len(tokenizer.encode(base_prompt))
prompt = (base_prompt, base_prompt_len)

# Create a client for spawning requests
clients = [OpenAIChatCompletionsClient.remote()]

req_launcher = RequestsLauncher(clients)

req_config = RequestConfig(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt=prompt
    )

req_launcher.launch_requests(req_config)
result = req_launcher.get_next_ready(block=True)
print(result)

```

# Implementing New LLM Clients

To implement a new LLM client, you need to implement the base class `llmperf.ray_llm_client.LLMClient` and decorate it as a ray actor.

```python

from llmperf.ray_llm_client import LLMClient
import ray


@ray.remote
class CustomLLMClient(LLMClient):

    def llm_request(self, request_config: RequestConfig) -> Tuple[Metrics, str, RequestConfig]:
        """Make a single completion request to a LLM API

        Returns:
            Metrics about the performance charateristics of the request.
            The text generated by the request to the LLM API.
            The request_config used to make the request. This is mainly for logging purposes.

        """
        ...

```

# Legacy Codebase
The old LLMPerf code base can be found in the [llmperf-legacy](https://github.com/ray-project/llmval-legacy) repo.
