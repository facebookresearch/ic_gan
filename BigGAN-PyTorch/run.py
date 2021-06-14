import utils
from trainer import run
from submitit.helpers import Checkpointable
import submitit
import json



class Trainer(Checkpointable):
    def __call__(self, config):
        if config['run_setup'] == 'local_debug':
            run(config, 'local_debug')
        else:
            run(config, 'slurm', master_node =submitit.JobEnvironment().hostnames[0])

if __name__ == "__main__":

    parser = utils.prepare_parser()
    config = vars(parser.parse_args())

    if config['json_config'] != "":
        data = json.load(open(config['json_config']))
    else:
        raise ValueError('Need config file!')
    for key in data.keys():
        config[key] = data[key]
    config['G_batch_size'] = config['batch_size']
    config['batch_size'] = config['batch_size'] * config['num_D_accumulations'] * \
                           config['num_D_steps']


    # config['G_shared'] = True
    # if config['instance_cond']:
    #     config['G_shared_feat'] = True

    trainer = Trainer()
    # import numpy as np
    # np.save('new_dict_debug', config)
    if config['run_setup'] == 'local_debug':
        trainer(config)
    else:
        print('Using ', config['n_nodes'], ' nodes and ', config['n_gpus_per_node'], ' GPUs per node.')
        executor = submitit.SlurmExecutor(
            folder='/checkpoint/acasanova/submitit_logs_biggan',
            max_num_timeout=60)
        print('testing  DDP code')
        executor.update_parameters(
            gpus_per_node=config['n_gpus_per_node'], partition='learnlab', constraint='volta32gb',
         #   comment='NeurIPS deadline',
            nodes=config['n_nodes'], ntasks_per_node=config['n_gpus_per_node'],
            cpus_per_task=8,
            mem=256000,
            time=3200, job_name=config['experiment_name'],
            exclusive=True if config['n_gpus_per_node']==8 else False)

        executor.submit(trainer, config)
        import time
        time.sleep(1)
