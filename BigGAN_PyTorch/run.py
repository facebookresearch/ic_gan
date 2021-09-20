# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import utils
from trainer import run
from submitit.helpers import Checkpointable

LOCAL = False
try:
    import submitit
except:
    print(
        "No submitit package found! Defaulting to executing the script in the local machine"
    )
    LOCAL = True
import json


class Trainer(Checkpointable):
    def __call__(self, config):
        if config["run_setup"] == "local_debug" or LOCAL:
            run(config, "local_debug")
        else:
            run(config, "slurm", master_node=submitit.JobEnvironment().hostnames[0])


if __name__ == "__main__":
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())

    if config["json_config"] != "":
        data = json.load(open(config["json_config"]))
        for key in data.keys():
            config[key] = data[key]
    else:
        print("Not using JSON configuration file!")
    config["G_batch_size"] = config["batch_size"]
    config["batch_size"] = (
        config["batch_size"] * config["num_D_accumulations"] * config["num_D_steps"]
    )

    trainer = Trainer()
    if config["run_setup"] == "local_debug" or LOCAL:
        trainer(config)
    else:
        print(
            "Using ",
            config["n_nodes"],
            " nodes and ",
            config["n_gpus_per_node"],
            " GPUs per node.",
        )
        executor = submitit.SlurmExecutor(
            folder=config["slurm_logdir"], max_num_timeout=60
        )
        executor.update_parameters(
            gpus_per_node=config["n_gpus_per_node"],
            partition=config["partition"],
            constraint="volta32gb",
            nodes=config["n_nodes"],
            ntasks_per_node=config["n_gpus_per_node"],
            cpus_per_task=8,
            mem=256000,
            time=3200,
            job_name=config["experiment_name"],
            exclusive=True if config["n_gpus_per_node"] == 8 else False,
        )

        executor.submit(trainer, config)
        import time

        time.sleep(1)
