from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tokenizers import (
    pre_tokenizers,
    decoders,
    Tokenizer,
    Regex
)
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

import numpy as np

ds = load_dataset("bigcode/the-stack-march-sample-special-tokens-stripped", split="train")
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<fim_pad>",
    "<filename>",
    "<gh_stars>",
    "<issue_start>",
    "<issue_comment>",
    "<issue_closed>",
    "<jupyter_start>",
    "<jupyter_text>",
    "<jupyter_code>",
    "<jupyter_output>",
    "<empty_output>",
    "<commit_before>",
    "<commit_msg>",
    "<commit_after>",
    "<reponame>",
]

def batch_iterator(dataset, batch_size=1000, column="content"):
    batches = np.array_split(range(len(dataset)), len(dataset)/batch_size) # bit slow.
    for batch in batches:
        yield dataset.select(batch)[column]
VOCAB_SIZE = 49_152

digits_pretokenizer = pre_tokenizers.Digits(individual_digits=True)
w_1_regex = Regex(r"""(\S+\s*\S+)""")
w_1_pretokenizer = pre_tokenizers.Split(pattern=w_1_regex, behavior="removed", invert=True)
bytelevel_pretokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)

bytelevel_decoder = decoders.ByteLevel()

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = pre_tokenizers.Sequence([w_1_pretokenizer, digits_pretokenizer, bytelevel_pretokenizer])
tokenizer.decoder = bytelevel_decoder

trainer = BpeTrainer(vocab_size=VOCAB_SIZE, show_progress=True, special_tokens=SPECIAL_TOKENS)

tokenizer.train_from_iterator(batch_iterator(ds), trainer=trainer)

tokenizer_wrapper = GPT2TokenizerFast(
    tokenizer_object=tokenizer,
    vocab_size=VOCAB_SIZE,
    additional_special_tokens=SPECIAL_TOKENS,
    bos_token=SPECIAL_TOKENS[0],
    eos_token=SPECIAL_TOKENS[0],
    unk_token=SPECIAL_TOKENS[0]
)

tokenizer_wrapper.push_to_hub("bigcode/tokenizer-ablations", revision="1-whitespace")