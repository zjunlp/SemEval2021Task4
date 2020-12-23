# coding=utf-8
import glob
import torch
from transformers.modeling_auto import AutoModel
import logging
from logging import debug
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    set_seed,
    LongformerModel,
    RobertaForMultipleChoice,
    AlbertForMultipleChoice
)
from utils import MultipleChoiceDataset, Split, processors, MultipleChoiceSlidingDataset, TrainingArguments
from model import Trainer


logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    eval_all_checkpoints: bool = field(
        default=False, 
    )
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    if training_args.sliding_window:
        train_dataset = (
            MultipleChoiceSlidingDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                task=data_args.task_name,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.train,
            )
            if training_args.do_train
            else None
        )
        eval_dataset = (
            MultipleChoiceSlidingDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                task=data_args.task_name,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.dev,
            )
            if training_args.do_eval
            else None
        )

    else:
        train_dataset = (
            MultipleChoiceDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                task=data_args.task_name,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.train,
            )
            if training_args.do_train
            else None
        )
        eval_dataset = (
            MultipleChoiceDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                task=data_args.task_name,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.dev,
            )
            if training_args.do_eval
            else None
        )

    
    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    results['eval_acc'] = 0.0
    results['eval_loss'] = 100000.0 # inf
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        # result = trainer.evaluate()
        # results.update(result)
        # Evaluate
        global_step = 99
        # if trainer.is_world_master():
        #     with open(output_eval_file, "a") as writer:
        #         logger.info("***** Eval results *****  " + str(global_step))
        #         writer.write('\neeeevvvvaaaallll\n')
        #         writer.writelines('eval on ' + data_args.data_dir + '\n')
        #         for key, value in result.items():
        #             logger.info("  %s = %s", key, value)
        #             writer.write("%s = %s\n" % (key, value))

        if data_args.eval_all_checkpoints:
            logger.info("Loading checkpoints saved during training for evaluation")
            
            # prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            # choice0 = "It is eaten with a fork and a knife."
            # choice1 = "It is eaten while held in the hand."
            # choice2= "It is eaten  held in the hand."
            # choice3 = "It is  while held in the hand."
            # choice4 = "It is eaten while held in the ."
            # labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

            # encoding = tokenizer([[prompt, prompt, prompt, prompt, prompt], [choice0, choice1, choice2, choice3, choice4]], return_tensors='pt', padding=True)
            # outputs = model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels)  # batch size is 1

            #  # the linear classifier still needs to be trained
            # loss = outputs.loss
            # logits = outputs.logits
            # import IPython; IPython.embed(); exit(1)
            



            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(training_args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )

            best_model = 99
            best_acc = results['eval_acc']
            for checkpoint in checkpoints:
                # Reload the model
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""


                model = AutoModelForMultipleChoice.from_pretrained(checkpoint)  
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=compute_metrics,
                )
                result = trainer.evaluate()
                results.update(result)
                # Evaluate
                # if trainer.is_world_master():
                #     with open(output_eval_file, "a") as writer:
                #         logger.info("***** Eval results *****  " + str(global_step))
                #         for key, value in result.items():
                #             logger.info("  %s = %s", key, value)
                #             writer.write("%s = %s\n" % (key, value))

                if results['eval_acc'] > best_acc:
                    best_model = global_step
                    best_acc = results['eval_acc']
            if trainer.is_world_master():
                results['eval_acc'] = best_acc
                with open(output_eval_file, "a") as writer:
                    logger.info("***** Eval results *****  " + str(global_step))
                    writer.writelines('eval on task path' + data_args.data_dir + '\n')
                    for key, value in result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
    return results




if __name__ == "__main__":
    main()
