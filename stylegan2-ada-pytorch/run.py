""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

from train import main
from submitit.helpers import Checkpointable
import submitit
import parser


class Trainer(Checkpointable):
    def __call__(self,args, slurm=False):
     if slurm:
         main(args, args.outdir, master_node =submitit.JobEnvironment().hostnames[0], port=args.port)
     else:
         main(args, args.outdir, master_node='', dry_run=args.dry_run)



if __name__ == "__main__":
    parser_ = parser.get_parser()
    args = parser_.parse_args()

    trainer = Trainer()
    if not args.slurm:
        trainer(args)
    else:

        executor = submitit.SlurmExecutor(
            folder=args.slurm_logdir,
            max_num_timeout=60)
        print(args.gpus)
        executor.update_parameters(
            gpus_per_node=args.gpus, partition=args.partition, constraint='volta32gb',
            nodes=args.nodes,
            ntasks_per_node = args.gpus,
            cpus_per_task=10,
            mem=256000,
            time=args.slurm_time, job_name=args.exp_name,
            exclusive=True if args.gpus==8 else False)

        job = executor.submit(trainer, args, slurm=True)
        print(job.job_id)

        import time
        time.sleep(1)