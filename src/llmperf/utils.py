import json
import math
import pathlib
import random
import subprocess
import time
from typing import Any, Dict, Tuple

from transformers import LlamaTokenizerFast


RESULTS_VERSION = "2025-10-14"


class LLMPerfResults:
    def __init__(
        self,
        name: str,
        metadata: Dict[str, Any] = None,
    ):
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata["timestamp"] = self.timestamp
        self.version = RESULTS_VERSION

    def to_dict(self):
        data = {
            "version": self.version,
            "name": self.name,
        }
        data.update(self.metadata)
        data = flatten_dict(data)
        return data

    def json(self):
        data = self.to_dict()
        return json.dumps(data)


def upload_to_s3(results_path: str, s3_path: str) -> None:
    """Upload the results to s3.

    Args:
        results_path: The path to the results file.
        s3_path: The s3 path to upload the results to.

    """

    command = ["aws", "s3", "sync", results_path, f"{s3_path}/"]
    result = subprocess.run(command)
    if result.returncode == 0:
        print("Files uploaded successfully!")
    else:
        print("An error occurred:")
        print(result.stderr)


# Globals for caching
SONNET_LINES = []
SONNET_TOKENS = []   # tokenized version of each line
SONNET_TOKEN_COUNTS = []   # precomputed lengths


def preload_sonnet(tokenizer):
    """
    Preload Shakespeare sonnet lines and their tokenizations into memory.

    Args:
        tokenizer: HuggingFace tokenizer.
    """
    global SONNET_LINES, SONNET_TOKENS, SONNET_TOKEN_COUNTS

    if SONNET_LINES:  # already loaded
        return

    sonnet_path = pathlib.Path(__file__).parent.resolve() / "sonnet.txt"
    with open(sonnet_path, "r", encoding="utf-8") as f:
        SONNET_LINES = f.readlines()

    SONNET_TOKENS = [tokenizer.encode(line) for line in SONNET_LINES]
    SONNET_TOKEN_COUNTS = [len(tokens) for tokens in SONNET_TOKENS]


def build_scheduled_sonnet_prompt(
        input_tokens: int,
        output_tokens: int,
        tokenizer,
) -> Tuple[str, int]:
    """
    Build a prompt of exactly `input_tokens` tokens using preloaded
    Shakespeare sonnet lines. Matches the signature of
    randomly_sample_sonnet_lines_prompt for drop-in replacement.

    Args:
        input_tokens: Desired number of input tokens (from schedule).
        output_tokens: Desired number of output tokens (from schedule).
        tokenizer: HuggingFace tokenizer (preloaded).

    Returns:
        Tuple[str, int]: Prompt string and token count.
    """
    global SONNET_LINES, SONNET_TOKENS, SONNET_TOKEN_COUNTS

    if not SONNET_LINES:
        preload_sonnet(tokenizer)

    # Base text
    base_prompt = (
        "Randomly stream lines from the following text "
        f"with {output_tokens} output tokens. "
        "Don't generate eos tokens:\n\n"
    )
    base_tokens = len(tokenizer.encode(base_prompt))

    if input_tokens < base_tokens:
        raise ValueError(
            f"Requested input_tokens={input_tokens} is too small; "
            f"needs at least {base_tokens}"
        )

    remaining = input_tokens - base_tokens
    prompt_parts = [base_prompt]
    total_tokens = base_tokens

    # Shuffle once per request (indexes, not strings)
    idxs = list(range(len(SONNET_LINES)))
    random.shuffle(idxs)

    # Extend the index list until it can cover the remaining tokens
    total_sonnet_tokens = sum(SONNET_TOKEN_COUNTS)
    while total_sonnet_tokens < remaining:
        extra = list(range(len(SONNET_LINES)))
        random.shuffle(extra)
        idxs.extend(extra)
        total_sonnet_tokens += sum(SONNET_TOKEN_COUNTS)

    for i in idxs:
        if remaining <= 0:
            break

        line, tokens, line_tokens = (
            SONNET_LINES[i],
            SONNET_TOKENS[i],
            SONNET_TOKEN_COUNTS[i],
        )

        if line_tokens > remaining:
            # Truncate safely by tokens
            truncated = tokens[:remaining]
            prompt_parts.append(tokenizer.decode(truncated))
            total_tokens += len(truncated)
            remaining = 0
            break

        # Take whole line
        prompt_parts.append(line)
        remaining -= line_tokens
        total_tokens += line_tokens

    return "".join(prompt_parts), total_tokens #TODO: use the tokenizer for a final verification and adjust. Due to line endings causing tokenization differences.


def randomly_sample_sonnet_lines_prompt(
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 250,
    expect_output_tokens: int = 150,
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer")
) -> Tuple[str, int]:
    """Generate a prompt that randomly samples lines from a the shakespeare sonnet at sonnet.txt.

    Args:
        prompt_length_mean: The mean length of the prompt to generate.
        prompt_len_stddev: The standard deviation of the length of the prompt to generate.
        expect_output_tokens: The number of tokens to expect in the output. This is used to
        determine the length of the prompt. The prompt will be generated such that the output
        will be approximately this many tokens.

    Note:
        tokens will be counted from the sonnet using the Llama tokenizer. Using one tokenizer
        ensures a fairer comparison across different LLMs. For example, if gpt 3.5 tokenizes
        a prompt in less tokens than Llama2, then this will be reflected in the results since
        they will be fed identical prompts.

    Returns:
        A tuple of the prompt and the length of the prompt.
    """

    get_token_length = lambda text: len(tokenizer.encode(text))

    prompt = (
        "Randomly stream lines from the following text "
        f"with {expect_output_tokens} output tokens. "
        "Don't generate eos tokens:\n\n"
    )
    # get a prompt length that is at least as long as the base
    num_prompt_tokens = sample_random_positive_int(
        prompt_tokens_mean, prompt_tokens_stddev
    )
    while num_prompt_tokens < get_token_length(prompt):
        num_prompt_tokens = sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )
    remaining_prompt_tokens = num_prompt_tokens - get_token_length(prompt)
    sonnet_path = pathlib.Path(__file__).parent.resolve() / "sonnet.txt"
    with open(sonnet_path, "r") as f:
        sonnet_lines = f.readlines()
    random.shuffle(sonnet_lines)
    sampling_lines = True
    while sampling_lines:
        for line in sonnet_lines:
            line_to_add = line
            if remaining_prompt_tokens - get_token_length(line_to_add) < 0:
                # This will cut off a line in the middle of a word, but that's ok since an
                # llm should be able to handle that.
                line_to_add = line_to_add[: int(math.ceil(remaining_prompt_tokens))]
                sampling_lines = False
                prompt += line_to_add
                break
            prompt += line_to_add
            remaining_prompt_tokens -= get_token_length(line_to_add)
    return (prompt, num_prompt_tokens)


def sample_random_positive_int(mean: int, stddev: int) -> int:
    """Sample random numbers from a gaussian distribution until a positive number is sampled.

    Args:
        mean: The mean of the gaussian distribution to sample from.
        stddev: The standard deviation of the gaussian distribution to sample from.

    Returns:
        A random positive integer sampled from the gaussian distribution.
    """
    ret = -1
    while ret <= 0:
        ret = int(random.gauss(mean, stddev))
    return ret


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
