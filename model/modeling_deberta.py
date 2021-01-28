import math
from collections.abc import Sequence

import torch
from packaging import version
from torch import _softmax_backward_data, nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    MultipleChoiceModelOutput
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

import sys
sys.path.append("..")
from utils import LabelSmoothingLoss

import os
import pathlib
import random
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple
from zipfile import ZipFile

import tqdm

import requests

from transformers.tokenization_utils import PreTrainedTokenizer


try:
    import regex as re
except ImportError:
    raise ImportError("Please install regex with: pip install regex")


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "bpe_encoder.bin"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/bpe_encoder.bin",
        "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/bpe_encoder.bin",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-base": 512,
    "microsoft/deberta-large": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-base": {"do_lower_case": False},
    "microsoft/deberta-large": {"do_lower_case": False},
}

__all__ = ["DebertaTokenizer"]


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings. The reversible bpe codes work on unicode
    strings. This means you need a large # of unicode characters in your vocab if you want to avoid UNKs. When you're
    at something like a 10B token dataset you end up needing around 5K for decent coverage. This is a signficant
    percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and unicode
    strings. And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word. Word is represented as tuple of symbols (symbols being variable-length
    strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip([tuple(k) for k in bpe_merges], range(len(bpe_merges))))
        self.cache = {}
        self.random = random.Random(0)

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def split_to_words(self, text):
        return list(re.findall(self.pat, text))

    def encode(self, text):
        bpe_tokens = []
        for token in self.split_to_words(text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text


def get_encoder(encoder, vocab):
    return Encoder(
        encoder=encoder,
        bpe_merges=vocab,
    )


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def download_asset(name, tag=None, no_cache=False, cache_dir=None):
    _tag = tag
    if _tag is None:
        _tag = "latest"
    if not cache_dir:
        cache_dir = os.path.join(pathlib.Path.home(), f".~DeBERTa/assets/{_tag}/")
    os.makedirs(cache_dir, exist_ok=True)
    output = os.path.join(cache_dir, name)
    if os.path.exists(output) and (not no_cache):
        return output

    repo = "https://api.github.com/repos/microsoft/DeBERTa/releases"
    releases = requests.get(repo).json()
    if tag and tag != "latest":
        release = [r for r in releases if r["name"].lower() == tag.lower()]
        if len(release) != 1:
            raise Exception(f"{tag} can't be found in the repository.")
    else:
        release = releases[0]
    asset = [s for s in release["assets"] if s["name"].lower() == name.lower()]
    if len(asset) != 1:
        raise Exception(f"{name} can't be found in the release.")
    url = asset[0]["url"]
    headers = {}
    headers["Accept"] = "application/octet-stream"
    resp = requests.get(url, stream=True, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Request for {url} return {resp.status_code}, {resp.text}")
    try:
        with open(output, "wb") as fs:
            progress = tqdm(
                total=int(resp.headers["Content-Length"]) if "Content-Length" in resp.headers else -1,
                ncols=80,
                desc=f"Downloading {name}",
            )
            for c in resp.iter_content(chunk_size=1024 * 1024):
                fs.write(c)
            progress.update(len(c))
            progress.close()
    except Exception:
        os.remove(output)
        raise

    return output


def load_vocab(name=None, tag=None, no_cache=False, cache_dir=None):
    import torch

    if name is None:
        name = "bpe_encoder"

    model_path = name
    if model_path and (not os.path.exists(model_path)) and not (("/" in model_path) or ("\\" in model_path)):
        _tag = tag
        if _tag is None:
            _tag = "latest"
        if not cache_dir:
            cache_dir = os.path.join(pathlib.Path.home(), f".~DeBERTa/assets/{_tag}/")
        os.makedirs(cache_dir, exist_ok=True)
        out_dir = os.path.join(cache_dir, name)
        model_path = os.path.join(out_dir, "bpe_encoder.bin")
        if (not os.path.exists(model_path)) or no_cache:
            asset = download_asset(name + ".zip", tag=tag, no_cache=no_cache, cache_dir=cache_dir)
            with ZipFile(asset, "r") as zipf:
                for zip_info in zipf.infolist():
                    if zip_info.filename[-1] == "/":
                        continue
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zipf.extract(zip_info, out_dir)
    elif not model_path:
        return None, None

    encoder_state = torch.load(model_path)
    return encoder_state


class GPT2Tokenizer(object):
    """
    A wrapper of GPT2 tokenizer with similar interface as BERT tokenizer
    Args:
        vocab_file (:obj:`str`, optional):
            The local path of vocabulary package or the release name of vocabulary in `DeBERTa GitHub releases
            <https://github.com/microsoft/DeBERTa/releases>`_, e.g. "bpe_encoder", default: `None`.
            If it's `None`, then it will download the vocabulary in the latest release from GitHub. The vocabulary file
            is a state dictionary with three items, "dict_map", "vocab", "encoder" which correspond to three files used
            in `RoBERTa`, i.e. `dict.txt`, `vocab.txt` and `encoder.json`. The difference between our wrapped GPT2
            tokenizer and RoBERTa wrapped tokenizer are,
            - Special tokens, unlike `RoBERTa` which use `<s>`, `</s>` as the `start` token and `end` token of a
              sentence. We use `[CLS]` and `[SEP]` as the `start` and `end` token of input sentence which is the same
              as `BERT`.
            - We remapped the token ids in our dictionary with regarding to the new special tokens, `[PAD]` => 0,
              `[CLS]` => 1, `[SEP]` => 2, `[UNK]` => 3, `[MASK]` => 50264
        special_tokens (:obj:`list`, optional):
            List of special tokens to be added to the end of the vocabulary.
    """

    def __init__(self, vocab_file=None, special_tokens=None):
        self.pad_token = "[PAD]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"

        self.symbols = []
        self.count = []
        self.indices = {}
        self.pad_token_id = self.add_symbol(self.pad_token)
        self.cls_token_id = self.add_symbol(self.cls_token)
        self.sep_token_id = self.add_symbol(self.sep_token)
        self.unk_token_id = self.add_symbol(self.unk_token)

        self.gpt2_encoder = load_vocab(vocab_file)
        self.bpe = get_encoder(self.gpt2_encoder["encoder"], self.gpt2_encoder["vocab"])
        for w, n in self.gpt2_encoder["dict_map"]:
            self.add_symbol(w, n)

        self.mask_token = "[MASK]"
        self.mask_id = self.add_symbol(self.mask_token)
        self.special_tokens = ["[MASK]", "[SEP]", "[PAD]", "[UNK]", "[CLS]"]
        if special_tokens is not None:
            for t in special_tokens:
                self.add_special_token(t)

        self.vocab = self.indices
        self.ids_to_tokens = self.symbols

    def tokenize(self, text):
        """
        Convert an input text to tokens.
        Args:
          text (:obj:`str`): input text to be tokenized.
        Returns:
          A list of byte tokens where each token represent the byte id in GPT2 byte dictionary
        Example::
          >>> tokenizer = GPT2Tokenizer()
          >>> text = "Hello world!"
          >>> tokens = tokenizer.tokenize(text)
          >>> print(tokens)
          ['15496', '995', '0']
        """
        bpe = self._encode(text)

        return [t for t in bpe.split(" ") if t]

    def convert_tokens_to_ids(self, tokens):
        """
        Convert list of tokens to ids
        Args:
          tokens (:obj:`list<str>`): list of tokens
        Returns:
          List of ids
        """

        return [self.vocab[t] for t in tokens]

    def convert_ids_to_tokens(self, ids):
        """
        Convert list of ids to tokens
        Args:
          ids (:obj:`list<int>`): list of ids
        Returns:
          List of tokens
        """

        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def split_to_words(self, text):
        return self.bpe.split_to_words(text)

    def decode(self, tokens):
        """
        Decode list of tokens to text strings
        Args:
          tokens (:obj:`list<str>`): list of tokens.
        Returns:
          Text string corresponds to the input tokens.
        Example::
          >>> tokenizer = GPT2Tokenizer()
          >>> text = "Hello world!"
          >>> tokens = tokenizer.tokenize(text)
          >>> print(tokens)
          ['15496', '995', '0']
          >>> tokenizer.decode(tokens)
          'Hello world!'
        """
        return self.bpe.decode([int(t) for t in tokens if t not in self.special_tokens])

    def add_special_token(self, token):
        """
        Adds a special token to the dictionary
        Args:
          token (:obj:`str`): Tthe new token/word to be added to the vocabulary.
        Returns:
          The id of new token in the vocabulary.
        """
        self.special_tokens.append(token)
        return self.add_symbol(token)

    def part_of_whole_word(self, token, is_bos=False):
        if is_bos:
            return True
        s = self._decode(token)
        if len(s) == 1 and (_is_whitespace(list(s)[0]) or _is_control(list(s)[0]) or _is_punctuation(list(s)[0])):
            return False

        return not s.startswith(" ")

    def sym(self, id):
        return self.ids_to_tokens[id]

    def id(self, sym):
        return self.vocab[sym]

    def _encode(self, x: str) -> str:
        return " ".join(map(str, self.bpe.encode(x)))

    def _decode(self, x: str) -> str:
        return self.bpe.decode(map(int, x.split()))

    def add_symbol(self, word, n=1):
        """
        Adds a word to the dictionary
        Args:
          word (:obj:`str`): Tthe new token/word to be added to the vocabulary.
          n (int, optional): The frequency of the word.
        Returns:
          The id of the new word.
        """
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def save_pretrained(self, path: str, filename_prefix: str = None):
        import torch

        filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
        if filename_prefix is not None:
            filename = filename_prefix + "-" + filename
        full_path = os.path.join(path, filename)
        torch.save(self.gpt2_encoder, full_path)
        return (full_path,)


class DebertaTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a DeBERTa tokenizer, which runs end-to-end tokenization: punctuation splitting + wordpiece
    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = XxxTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.do_lower_case = do_lower_case
        self.gpt2_tokenizer = GPT2Tokenizer(vocab_file)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab(self):
        return self.gpt2_tokenizer.vocab

    def get_vocab(self):
        vocab = self.vocab.copy()
        vocab.update(self.get_added_vocab())
        return vocab

    def _tokenize(self, text):
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        if self.do_lower_case:
            text = text.lower()
        return self.gpt2_tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.gpt2_tokenizer.sym(index) if index < self.vocab_size else self.unk_token

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        return self.gpt2_tokenizer.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:
        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(
                    lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0,
                )
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
        sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return self.gpt2_tokenizer.save_pretrained(save_directory, filename_prefix=filename_prefix)

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/config.json",
    "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/config.json",
}


class DebertaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.DebertaModel` or a
    :class:`~transformers.TFDebertaModel`. It is used to instantiate a DeBERTa model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the DeBERTa `microsoft/deberta-base <https://huggingface.co/microsoft/deberta-base>`__
    architecture.
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.
    Arguments:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the DeBERTa model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.DebertaModel` or
            :class:`~transformers.TFDebertaModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"`, :obj:`"gelu"`, :obj:`"tanh"`, :obj:`"gelu_fast"`,
            :obj:`"mish"`, :obj:`"linear"`, :obj:`"sigmoid"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.DebertaModel` or
            :class:`~transformers.TFDebertaModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        relative_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether use relative position encoding.
        max_relative_positions (:obj:`int`, `optional`, defaults to 1):
            The range of relative positions :obj:`[-max_position_embeddings, max_position_embeddings]`. Use the same
            value as :obj:`max_position_embeddings`.
        pad_token_id (:obj:`int`, `optional`, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether add absolute position embedding to content embedding.
        pos_att_type (:obj:`List[str]`, `optional`):
            The type of relative position attention, it can be a combination of :obj:`["p2c", "c2p", "p2p"]`, e.g.
            :obj:`["p2c"]`, :obj:`["p2c", "c2p"]`, :obj:`["p2c", "c2p", 'p2p"]`.
        layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    """
    model_type = "deberta"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=False,
        max_relative_positions=-1,
        pad_token_id=0,
        position_biased_input=True,
        pos_att_type=None,
        pooler_dropout=0,
        pooler_hidden_act="gelu",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input

        # Backwards compatibility
        if type(pos_att_type) == str:
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DebertaConfig"
_TOKENIZER_FOR_DOC = "DebertaTokenizer"

DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/deberta-base",
    "microsoft/deberta-large",
]


class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states, mask=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory
    Args:
      input (:obj:`torch.tensor`): The input tensor that will apply softmax.
      mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
      dim (int): The dimension that will apply softmax
    Example::
      import torch
      from transformers.models.deberta import XSoftmax
      # Make a tensor
      x = torch.randn([4,20,100])
      # Create a mask
      mask = (x>0).int()
      y = XSoftmax.apply(x, mask, dim=-1)
    """

    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        if version.Version(torch.__version__) >= version.Version("1.2.0a"):
            rmask = ~(mask.bool())
        else:
            rmask = (1 - mask).byte()  # This line is not supported by Onnx tracing.

        output = input.masked_fill(rmask, float("-inf"))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        (output,) = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        if version.Version(torch.__version__) >= version.Version("1.2.0a"):
            mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()
        else:
            mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).byte()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(torch.nn.Module):
    """
    Optimized dropout module for stabilizing the training
    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module
        Args:
            x (:obj:`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


class DebertaLayerNorm(nn.Module):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_type = hidden_states.dtype
        hidden_states = hidden_states.float()
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.to(input_type)
        y = self.weight * hidden_states + self.bias
        return y


class DebertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaSelfOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        self_output = self.self(
            hidden_states,
            attention_mask,
            return_att,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if return_att:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)

        if return_att:
            return (attention_output, att_matrix)
        else:
            return attention_output


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Deberta
class DebertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DebertaOutput(nn.Module):
    def __init__(self, config):
        super(DebertaOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaLayer(nn.Module):
    def __init__(self, config):
        super(DebertaLayer, self).__init__()
        self.attention = DebertaAttention(config)
        self.intermediate = DebertaIntermediate(config)
        self.output = DebertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            return_att=return_att,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if return_att:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if return_att:
            return (layer_output, att_matrix)
        else:
            return layer_output


class DebertaEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([DebertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, config.hidden_size)

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), hidden_states.device)
        return relative_pos

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(
                next_kv,
                attention_mask,
                output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )
            if output_attentions:
                hidden_states, att_m = hidden_states

            if query_states is not None:
                query_states = hidden_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


def build_relative_position(query_size, key_size, device):
    """
    Build relative position according to the query and key
    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key
    :math:`P_k` is range from (0, key_size), The relative positions from query to key is :math:`R_{q \\rightarrow k} =
    P_q - P_k`
    Args:
        query_size (int): the length of query
        key_size (int): the length of key
    Return:
        :obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]
    """

    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


class DisentangledSelfAttention(torch.nn.Module):
    """
    Disentangled self-attention module
    Parameters:
        config (:obj:`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            `BertConfig`, for more details, please refer :class:`~transformers.DebertaConfig`
    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.in_proj = torch.nn.Linear(config.hidden_size, self.all_head_size * 3, bias=False)
        self.q_bias = torch.nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.v_bias = torch.nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []

        self.relative_attention = getattr(config, "relative_attention", False)
        self.talking_head = getattr(config, "talking_head", False)

        if self.talking_head:
            self.head_logits_proj = torch.nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)
            self.head_weights_proj = torch.nn.Linear(
                config.num_attention_heads, config.num_attention_heads, bias=False
            )

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
                self.pos_proj = torch.nn.Linear(config.hidden_size, self.all_head_size, bias=False)
            if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
                self.pos_q_proj = torch.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Call the module
        Args:
            hidden_states (:obj:`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                `Attention(Q,K,V)`
            attention_mask (:obj:`torch.ByteTensor`):
                An attention mask matrix of shape [`B`, `N`, `N`] where `B` is the batch size, `N` is the maximum
                sequence length in which element [i,j] = `1` means the `i` th token in the input can attend to the `j`
                th token.
            return_att (:obj:`bool`, optional):
                Whether return the attention matrix.
            query_states (:obj:`torch.FloatTensor`, optional):
                The `Q` state in `Attention(Q,K,V)`.
            relative_pos (:obj:`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [`B`, `N`, `N`] with
                values ranging in [`-max_relative_positions`, `max_relative_positions`].
            rel_embeddings (:obj:`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [:math:`2 \\times
                \\text{max_relative_positions}`, `hidden_size`].
        """
        if query_states is None:
            qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
            query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
        else:

            def linear(w, b, x):
                if b is not None:
                    return torch.matmul(x, w.t()) + b.t()
                else:
                    return torch.matmul(x, w.t())  # + b.t()

            ws = self.in_proj.weight.chunk(self.num_attention_heads * 3, dim=0)
            qkvw = [torch.cat([ws[i * 3 + k] for i in range(self.num_attention_heads)], dim=0) for k in range(3)]
            qkvb = [None] * 3

            q = linear(qkvw[0], qkvb[0], query_states)
            k, v = [linear(qkvw[i], qkvb[i], hidden_states) for i in range(1, 3)]
            query_layer, key_layer, value_layer = [self.transpose_for_scores(x) for x in [q, k, v]]

        query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1 + len(self.pos_att_type)
        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        query_layer = query_layer / scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = attention_scores + rel_att

        # bxhxlxd
        if self.talking_head:
            attention_scores = self.head_logits_proj(attention_scores.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        if self.talking_head:
            attention_probs = self.head_weights_proj(attention_probs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_att:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[
            self.max_relative_positions - att_span : self.max_relative_positions + att_span, :
        ].unsqueeze(0)
        if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)

        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            score += c2p_att

        # position->content
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1) * scale_factor)
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if "p2c" in self.pos_att_type:
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(
                p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)
            ).transpose(-1, -2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            score += p2c_att

        return score


class DebertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.output_to_half = False
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DebertaConfig
    base_model_prefix = "deberta"
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


DEBERTA_START_DOCSTRING = r"""
    The DeBERTa model was proposed in `DeBERTa: Decoding-enhanced BERT with Disentangled Attention
    <https://arxiv.org/abs/2006.03654>`_ by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build on top of
    BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.```
    Parameters:
        config (:class:`~transformers.DebertaConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

DEBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`transformers.DebertaTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    DEBERTA_START_DOCSTRING,
)
class DebertaModel(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = DebertaEmbeddings(config)
        self.encoder = DebertaEncoder(config)
        self.z_steps = 0
        self.config = config
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    # @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="microsoft/deberta-base",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    return_att=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top. """, DEBERTA_START_DOCSTRING)
class DebertaForMaskedLM(DebertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        self.cls = DebertaOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="microsoft/deberta-base",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# copied from transformers.models.bert.BertPredictionHeadTransform with bert -> deberta
class DebertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# copied from transformers.models.bert.BertLMPredictionHead with bert -> deberta
class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = DebertaPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# copied from transformers.models.bert.BertOnlyMLMHead with bert -> deberta
class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = DebertaLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@add_start_docstrings(
    """
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForSequenceClassification(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="microsoft/deberta-base",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # regression task
                loss_fn = torch.nn.MSELoss()
                logits = logits.view(-1).to(labels.dtype)
                loss = loss_fn(logits, labels.view(-1))
            elif labels.dim() == 1 or labels.size(-1) == 1:
                label_index = (labels >= 0).nonzero()
                labels = labels.long()
                if label_index.size(0) > 0:
                    labeled_logits = torch.gather(logits, 0, label_index.expand(label_index.size(0), logits.size(1)))
                    labels = torch.gather(labels, 0, label_index.view(-1))
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                else:
                    loss = torch.tensor(0).to(logits)
            else:
                log_softmax = torch.nn.LogSoftmax(-1)
                loss = -((log_softmax(logits) * labels).sum(-1)).mean()
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        else:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


@add_start_docstrings(
    """
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForTokenClassification(DebertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="microsoft/deberta-base",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForQuestionAnswering(DebertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="microsoft/deberta-base",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class DebertaForMultipleChoiceWithLabelSmooth(DebertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.pooler = ContextPooler(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices-1]`` where :obj:`num_choices` is the size of the second dimension
            of the input tensors. (See :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        # flat the input are [label_size * batch_size, hidden_size]
        outputs = self.deberta(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = self.pooler(outputs[0])

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)



        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = LabelSmoothingLoss(num_choices, 0.1)
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # MultipleChoiceModelOutput a defined output for model
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
