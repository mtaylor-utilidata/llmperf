import asyncio
import csv
import logging
import argparse
import time
import gc
from pathlib import Path
from statistics import mean
from transformers import AutoTokenizer, LlamaTokenizerFast

# Optional: import tiktoken if available
try:
    import tiktoken
except ImportError:
    tiktoken = None

import httpx

from src.llmperf.utils import build_scheduled_sonnet_prompt


MAX_IN_FLIGHT = 512

# ------- Tokenizer helper constants ------

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


# ---------------- logging ----------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------- config ----------------

PREP_WINDOW_SECONDS = 10.0
FINAL_BUSY_WAIT = 0.001  # 1 ms

# ---------------- CSV loading ----------------

def load_schedule(csv_path: Path):
    rows = []

    with csv_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                delta = float(row[0])
                input_toks = int(row[1])
                output_toks = int(row[2])
            except (ValueError, IndexError):
                continue

            rows.append(
                {
                    "delta": delta,
                    "input_tokens": input_toks,
                    "output_tokens": output_toks,
                }
            )

    if not rows:
        raise ValueError("No valid schedule rows found")

    rows.sort(key=lambda r: r["delta"])
    return rows

# ---------------- warmup ----------------

async def warmup_connections(client, model, n=8):
    log.info("Warming up %d connections", n)
    dummy = {
        "model": model,
        "messages": [{"role": "user", "content": "warmup"}],
        "max_tokens": 1,
    }
    await asyncio.gather(
        *[client.post("/chat/completions", json=dummy) for _ in range(n)],
        return_exceptions=True,
    )
    log.info("Warmup complete")

# ---------------- prep manager ----------------

async def prep_manager(
        schedule,
        start_time,
        tokenizer,
        model,
        prepared,
        prep_times,
):
    """
    Prepares request bodies for all schedule entries that fall
    within the rolling prep window.
    """
    prompt_cache = {}
    idx = 0
    total = len(schedule)

    while idx < total:
        now = asyncio.get_running_loop().time()
        window_limit = now + PREP_WINDOW_SECONDS

        while idx < total and start_time + schedule[idx]["delta"] <= window_limit:
            row = schedule[idx]
            key = (row["input_tokens"], row["output_tokens"])

            prep_start = time.perf_counter()

            if key not in prompt_cache:
                prompt, _ = build_scheduled_sonnet_prompt(
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    tokenizer=tokenizer,
                )
                prompt_cache[key] = prompt

            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt_cache[key]}],
                "max_tokens": row["output_tokens"],
            }

            prep_dur = time.perf_counter() - prep_start
            prep_times.append(prep_dur)

            prepared[idx] = body
            log.debug(
                "prepared idx=%d input=%d output=%d prep_time=%.6f",
                idx,
                row["input_tokens"],
                row["output_tokens"],
                prep_dur,
            )

            idx += 1

        await asyncio.sleep(0.01)  # yield, don't spin

# ---------------- dispatch ----------------

async def fire(
        client,
        scheduled_t,
        idx,
        prepared,
        lags,
        conn_sem,
):
    loop = asyncio.get_running_loop()

    # Wait until prepared
    while idx not in prepared:
        await asyncio.sleep(0)

    body = prepared.pop(idx)

    # Final alignment (never early)
    while True:
        now = loop.time()
        if now >= scheduled_t - FINAL_BUSY_WAIT:
            break
        await asyncio.sleep(0)

    while loop.time() < scheduled_t:
        pass

    dispatch_t = loop.time()
    lag = dispatch_t - scheduled_t
    lags.append(lag)

    log.debug(
        "scheduled=%.9f dispatched=%.9f lag=%.9f",
        scheduled_t,
        dispatch_t,
        lag,
    )

    # --- bounded fire-and-forget ---
    async with conn_sem:
        try:
            async with client.stream(
                    "POST",
                    "/chat/completions",
                    json=body,
            ):
                pass
        except Exception as e:
            log.debug("dispatch failed idx=%d: %s", idx, e)


# ---------------- stats ----------------

async def lag_reporter(lags):
    while True:
        await asyncio.sleep(10)
        if not lags:
            continue
        log.info(
            "lag stats (n=%d): avg=%.6f min=%.6f max=%.6f",
            len(lags),
            sum(lags) / len(lags),
            min(lags),
            max(lags),
            )

def print_final_stats(lags, prep_times):
    print("\n" + "=" * 80)
    print("FINAL STATS")
    print("=" * 80)
    print(f"requests        : {len(lags)}")
    print(f"avg dispatch lag: {sum(lags)/len(lags):.9f}s")
    print(f"min dispatch lag: {min(lags):.9f}s")
    print(f"max dispatch lag: {max(lags):.9f}s")
    print(f"avg prep time   : {sum(prep_times)/len(prep_times):.6f}s")
    print("=" * 80 + "\n")

# ---------------- main ----------------

async def main(host, csv_path, api_key, model):
    schedule = load_schedule(csv_path)
    tokenizer = get_tokenizer_for_model(model, log=log)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    lags = []
    prep_times = []
    prepared = {}

    loop = asyncio.get_running_loop()
    start_time = loop.time() + 5

    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)

    gc.disable()

    async with httpx.AsyncClient(
            base_url=host,
            headers=headers,
            timeout=None,
            limits=limits,
            http2=False,
    ) as client:
        await warmup_connections(client, model)

        prep_task = asyncio.create_task(
            prep_manager(schedule, start_time, tokenizer, model, prepared, prep_times)
        )

        reporter = asyncio.create_task(lag_reporter(lags))

        conn_sem = asyncio.Semaphore(MAX_IN_FLIGHT)

        dispatch_tasks = [
            asyncio.create_task(
                fire(
                    client,
                    start_time + row["delta"],
                    idx,
                    prepared,
                    lags,
                    conn_sem
                    )
            )
            for idx, row in enumerate(schedule)
        ]

        await asyncio.gather(*dispatch_tasks)
        prep_task.cancel()
        reporter.cancel()

    gc.enable()
    print_final_stats(lags, prep_times)

# ------------ Get Tokenizer for model -----------
def get_tokenizer_for_model(model_name: str, log):
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
                log.info(f"Using known tokenizer for model '{model_name}': {key}")
                log.info(f"Tokenizer type: {type(tokenizer)}")
                _tokenizer_cache[model_name] = tokenizer
                return tokenizer
            else:
                log.warning(
                    f"Known tokenizer for '{model_name}' requires `tiktoken` but it’s not installed."
                )

    # 2. Hugging Face fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        log.info(f"Loaded HF tokenizer for model '{model_name}'.")
        _tokenizer_cache[model_name] = tokenizer
        return tokenizer
    except Exception as e:
        log.warning(
            f"Failed to load tokenizer for '{model_name}' from Hugging Face ({e}). "
            "Falling back to LlamaTokenizerFast."
        )

    # 3. Llama fallback
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    _tokenizer_cache[model_name] = tokenizer
    log.warning(f"USING THE FALLBACK 'LlamaTokenizerFast' TOKENIZER MAY RESULT IN INACCURATE TOKEN COUNTS WHEN USED WITH A MODEL WHICH USES A DIFFERENT TOKENIZER.")
    return tokenizer


# ---------------- entry ----------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--api-key", default="DUMMY")
    p.add_argument("--model", required=True)
    args = p.parse_args()

    asyncio.run(main(args.host, Path(args.csv), args.api_key, args.model))
