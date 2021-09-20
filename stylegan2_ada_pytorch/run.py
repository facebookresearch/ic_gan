# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from train import main
from submitit.helpers import Checkpointable

LOCAL = False
try:
    import submitit
except:
    print(
        "No submitit package found! Defaulting to executing the script in the local machine"
    )
    LOCAL = True
import parser
import json


class Trainer(Checkpointable):
    def __call__(self, args, slurm=False):
        if slurm and not LOCAL:
            main(
                args,
                args.outdir,
                master_node=submitit.JobEnvironment().hostnames[0],
                port=args.port,
            )
        else:
            main(args, args.outdir, master_node="", dry_run=args.dry_run)


if __name__ == "__main__":
    parser_ = parser.get_parser()
    args = parser_.parse_args()

    if args.json_config != "":
        data = json.load(open(args.json_config))
        for key in data.keys():
            setattr(args, key, data[key])
    else:
        print("Not using JSON configuration file!")
    if args.data_root is not None:
        print("Appending data_root to paths")
        args.data = os.path.join(args.data_root, args.data)
        args.root_feats = os.path.join(args.data_root, args.root_feats)
        args.root_nns = os.path.join(args.data_root, args.root_nns)
    args.outdir = args.base_root

    trainer = Trainer()
    if not args.slurm or LOCAL:
        trainer(args)
    else:

        executor = submitit.SlurmExecutor(folder=args.slurm_logdir, max_num_timeout=60)
        print(args.gpus)
        executor.update_parameters(
            gpus_per_node=args.gpus,
            partition=args.partition,
            constraint="volta32gb",
            nodes=args.nodes,
            ntasks_per_node=args.gpus,
            cpus_per_task=10,
            mem=256000,
            time=args.slurm_time,
            job_name=args.exp_name,
            exclusive=True if args.gpus == 8 else False,
        )

        job = executor.submit(trainer, args, slurm=True)
        print(job.job_id)

        import time

        time.sleep(1)
