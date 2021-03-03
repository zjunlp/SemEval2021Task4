# SemEval2021Task4

The 4th rank solution to [the SemEval 2021 Task4](https://competitions.codalab.org/competitions/26153)

## Environment

python >= 3.7

torch >= 1.6

transformers==3.3.1

## preprocess

To get the NAL answers, we use the models without fine-tuning to get the enhanced dataset.

```shell
python ./dataset/preprocess.py
```

## Training

### post-training

Firstly, pretrain the `ALBERT`,`RoBERTa` models to fit the in-domain text.

```sh
./scripts/pretrain_model.sh
```

### fine-tuning

Secondly, fine-tuning the model with the followed scripts.

```shell
./scripts/run_deberta.sh
./scripts/run_albert.sh
./scripts/run_roberta.sh
```

### ensemble

Finally, we get the best model files and ensemble them with weighted voting (weighted by the acc at dev set).

```shell
./scripts/get_answer/save_answer.sh
```



