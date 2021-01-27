# coding=utf-8
import logging
import json
from logging import debug
import os

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
    BertModel,
    AlbertForMultipleChoice,
)
from utils import MultipleChoiceDataset, Split, processors, MultipleChoiceSlidingDataset, TrainingArguments, ModelArguments, DataTrainingArguments
from model import Trainer, RobertaForMultipleChoiceWithLabelSmooth, AlbertForMultipleChoiceWithLabelSmooth, XLNetForMultipleChoiceWithLabelSmooth
from utils import delete_checkpoint_files_except_the_best, simple_accuracy, compute_metrics


logger = logging.getLogger(__name__)




def main():
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
        # label_list = processor.get_labels()
        # num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        # num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        mirror="tuna"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        mirror="tuna"
    )
    if training_args.label_smoothing:
        if "roberta" in model_args.model_name_or_path:
            model = RobertaForMultipleChoiceWithLabelSmooth.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        elif "albert" in model_args.model_name_or_path:
            model = AlbertForMultipleChoiceWithLabelSmooth.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        elif "xlnet" in model_args.model_name_or_path:
            model = XLNetForMultipleChoiceWithLabelSmooth.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        else:
            raise ValueError("model with smooth not implmented !")
    else:
        model = AutoModelForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            mirror="tuna"
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
            # model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        # It is not the best model, no need to save
        # trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("删除之前的checkpoint 文件， 保存最好的到当前目录")
        delete_checkpoint_files_except_the_best(training_args.output_dir)

        with open(os.path.join(training_args.output_dir, "log_history.json"), "r") as file:
            results = json.load(file)
            acc = results[-1]["eval_acc"]
        
        if acc >= 0.85:
            with open("result.txt", "a+") as file:
                file.writelines("acc :{}\n".format(acc))
                file.writelines("file path: {}\n".format(training_args.output_dir))
    if training_args.do_predict:
        #TODO 自动生成result.pkl
        pass

        
    results




if __name__ == "__main__":
    main()
