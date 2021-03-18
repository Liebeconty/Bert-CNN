from pathlib import Path


class Config(object):
    def __init__(self):
        self.CLAS_DATA_PATH = Path('RCV1')
        self.args = {
            "train_size": -1,
            "data_dir": 'RCV1/',
            "val_size": -1,
            "task_name": "toxic_multilabel",
            "no_cuda": False,
            # "no_cuda": True,
            "bert_model": 'bert-base-uncased',
            "max_seq_length": 512,
            "do_train": True,
            "do_eval": True,
            "do_lower_case": True,
            "train_batch_size": 4,
            "eval_batch_size": 8,
            "learning_rate": 3e-5,
            "num_train_epochs": 8.0,
            "warmup_proportion": 0.1,
            "local_rank": -1,
            "seed": 42,
            "gradient_accumulation_steps": 1,
            "optimize_on_cpu": False,
            "fp16": False,
            "loss_scale": 128,
            "output_dir": self.CLAS_DATA_PATH/'.output'
        }