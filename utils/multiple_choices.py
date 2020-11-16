# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """

import csv
import glob
import json
import logging
from logging import debug
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import tqdm

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MultipleChoiceDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            processor = processors[task]()

            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                    task,
                ),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    label_list = processor.get_labels()
                    if mode == Split.dev:
                        examples = processor.get_dev_examples(data_dir)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(data_dir)
                    else:
                        examples = processor.get_train_examples(data_dir)
                    logger.info("Training examples: %s", len(examples))
                    self.features = convert_examples_to_features(
                        examples,
                        label_list,
                        max_seq_length,
                        tokenizer,
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()



class SemEvalProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _read_json(self, input_file):
        res = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                json_f = json.loads(line)
                res.append(json_f)
        return res

    def _create_examples(self, lines: List[dict], type: str):
        """Creates examples for the training and dev sets."""

        examples = [
            InputExample(
                example_id= '111' + str(_),
                question= d['question'],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[d['article'],d['article'], d['article'], d['article'], d['article']],
                endings=[d['option_0'],d['option_1'],d['option_2'],d['option_3'],d['option_4']],
                label=str(d['label']),
            )
            for _, d in enumerate(lines)  # we skip the line with the column names
        ]

        return examples

    def fuzhi(self, sen):
        t = sen+ ' '
        return t * 5

class SemEvalEnhancedProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train_enhanced.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5"]

    def _read_json(self, input_file):
        res = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                json_f = json.loads(line)
                res.append(json_f)
        return res

    def _create_examples(self, lines: List[dict], type: str):
        """Creates examples for the training and dev sets."""

        if type == 'train':
            examples = [
                InputExample(
                    example_id= '111' + str(_),
                    question= d['question'],  # in the swag dataset, the
                    # common beginning of each
                    # choice is stored in "sent2".
                    contexts=[d['article']]*6 ,
                    endings=[d['option_0'],d['option_1'],d['option_2'],d['option_3'],d['option_4'], d['option_5']],
                    label=str(d['label']),
                )
                for _, d in enumerate(lines)  # we skip the line with the column names
            ]
        else:
            examples = [
                InputExample(
                    example_id= '111' + str(_),
                    question= d['question'],  # in the swag dataset, the
                    # common beginning of each
                    # choice is stored in "sent2".
                    contexts=[d['article']] * 6,
                    endings=[d['option_0'],d['option_1'],d['option_2'],d['option_3'],d['option_4'], ''],
                    label=str(d['label']),
                )
                for _, d in enumerate(lines)  # we skip the line with the column names
            ]

        return examples



# modify 
def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context

            if example.question.find("@placeholder") != -1:
                # this is for cloze question
                text_b = example.question.replace("@placeholder", ending)
            else:
                text_b = example.question + " " + ending

            # for bert like [text_b, text_a]
            # for xlnet [text_a[:400], text_b] seq_length 480
            inputs = tokenizer(
                text_b,
                text_a,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
                return_token_type_ids=True
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features

# def squad_convert_example_to_features(
#     example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
# ):
#     features = []
#     if is_training and not example.is_impossible:
#         # Get start and end position
#         start_position = example.start_position
#         end_position = example.end_position

#         # If the answer cannot be found in the text, then skip this example.
#         # actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
#         # cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
#         # if actual_text.find(cleaned_answer_text) == -1:
#         #     logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
#         #     return []

#     tok_to_orig_index = []
#     orig_to_tok_index = []
#     all_doc_tokens = []
#     for (i, token) in enumerate(example.doc_tokens):
#         # 这个复杂度是不是有点高啊， 完全可以用cnt+1
#         orig_to_tok_index.append(len(all_doc_tokens))
#         if tokenizer.__class__.__name__ in [
#             "RobertaTokenizer",
#             "LongformerTokenizer",
#             "BartTokenizer",
#             "RobertaTokenizerFast",
#             "LongformerTokenizerFast",
#             "BartTokenizerFast",
#         ]:
#             sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
#         else:
#             sub_tokens = tokenizer.tokenize(token)
#         for sub_token in sub_tokens:
#             tok_to_orig_index.append(i)
#             all_doc_tokens.append(sub_token)

#     if is_training and not example.is_impossible:
#         tok_start_position = orig_to_tok_index[example.start_position]
#         if example.end_position < len(example.doc_tokens) - 1:
#             tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
#         else:
#             tok_end_position = len(all_doc_tokens) - 1

#         (tok_start_position, tok_end_position) = _improve_answer_span(
#             all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
#         )

#     spans = []

#     truncated_query = tokenizer.encode(
#         example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
#     )

#     # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
#     # in the way they compute mask of added tokens.
#     tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
#     sequence_added_tokens = (
#         tokenizer.max_len - tokenizer.max_len_single_sentence + 1
#         if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
#         else tokenizer.max_len - tokenizer.max_len_single_sentence
#     )
#     sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

#     span_doc_tokens = all_doc_tokens
#     while len(spans) * doc_stride < len(all_doc_tokens):

#         # Define the side we want to truncate / pad and the text/pair sorting
#         if tokenizer.padding_side == "right":
#             texts = truncated_query
#             pairs = span_doc_tokens
#             truncation = TruncationStrategy.ONLY_SECOND.value
#         else:
#             texts = span_doc_tokens
#             pairs = truncated_query
#             truncation = TruncationStrategy.ONLY_FIRST.value

#         encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
#             texts,
#             pairs,
#             truncation=truncation,
#             padding=padding_strategy,
#             max_length=max_seq_length,
#             return_overflowing_tokens=True,
#             stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
#             return_token_type_ids=True,
#         )

#         paragraph_len = min(
#             len(all_doc_tokens) - len(spans) * doc_stride,
#             max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
#         )

#         if tokenizer.pad_token_id in encoded_dict["input_ids"]:
#             if tokenizer.padding_side == "right":
#                 non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
#             else:
#                 last_padding_id_position = (
#                     len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
#                 )
#                 non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

#         else:
#             non_padded_ids = encoded_dict["input_ids"]

#         tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

#         token_to_orig_map = {}
#         for i in range(paragraph_len):
#             index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
#             token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

#         encoded_dict["paragraph_len"] = paragraph_len
#         encoded_dict["tokens"] = tokens
#         encoded_dict["token_to_orig_map"] = token_to_orig_map
#         encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
#         encoded_dict["token_is_max_context"] = {}
#         encoded_dict["start"] = len(spans) * doc_stride
#         encoded_dict["length"] = paragraph_len

#         spans.append(encoded_dict)

#         if "overflowing_tokens" not in encoded_dict or (
#             "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
#         ):
#             break
#         span_doc_tokens = encoded_dict["overflowing_tokens"]

#     for doc_span_index in range(len(spans)):
#         for j in range(spans[doc_span_index]["paragraph_len"]):
#             is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
#             index = (
#                 j
#                 if tokenizer.padding_side == "left"
#                 else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
#             )
#             spans[doc_span_index]["token_is_max_context"][index] = is_max_context

#     for span in spans:
#         # Identify the position of the CLS token
#         cls_index = span["input_ids"].index(tokenizer.cls_token_id)

#         # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
#         # Original TF implem also keep the classification token (set to 0)
#         p_mask = np.ones_like(span["token_type_ids"])
#         if tokenizer.padding_side == "right":
#             p_mask[len(truncated_query) + sequence_added_tokens :] = 0
#         else:
#             p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

#         pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
#         special_token_indices = np.asarray(
#             tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
#         ).nonzero()

#         p_mask[pad_token_indices] = 1
#         p_mask[special_token_indices] = 1

#         # Set the cls index to 0: the CLS index can be used for impossible answers
#         p_mask[cls_index] = 0

#         span_is_impossible = example.is_impossible
#         start_position = 0
#         end_position = 0
#         if is_training and not span_is_impossible:
#             # For training, if our document chunk does not contain an annotation
#             # we throw it out, since there is nothing to predict.
#             doc_start = span["start"]
#             doc_end = span["start"] + span["length"] - 1
#             out_of_span = False

#             if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
#                 out_of_span = True

#             if out_of_span:
#                 start_position = cls_index
#                 end_position = cls_index
#                 span_is_impossible = True
#             else:
#                 if tokenizer.padding_side == "left":
#                     doc_offset = 0
#                 else:
#                     doc_offset = len(truncated_query) + sequence_added_tokens

#                 start_position = tok_start_position - doc_start + doc_offset
#                 end_position = tok_end_position - doc_start + doc_offset

#         features.append(
#             SquadFeatures(
#                 span["input_ids"],
#                 span["attention_mask"],
#                 span["token_type_ids"],
#                 cls_index,
#                 p_mask.tolist(),
#                 example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
#                 unique_id=0,
#                 paragraph_len=span["paragraph_len"],
#                 token_is_max_context=span["token_is_max_context"],
#                 tokens=span["tokens"],
#                 token_to_orig_map=span["token_to_orig_map"],
#                 start_position=start_position,
#                 end_position=end_position,
#                 is_impossible=span_is_impossible,
#                 qas_id=example.qas_id,
#             )
#         )
#     return features



processors = {"semeval" : SemEvalProcessor, 'semevalenhanced' : SemEvalEnhancedProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race", 4, "swag", 4, "arc", 4, "syn", 5, "semeval" , 5, "semevalenhanced", 6}