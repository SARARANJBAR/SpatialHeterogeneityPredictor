import os
import itertools as it

# define hyper parameter grid
# target could also be sox2 or cd68
# lambda_u is the contribution of unlabeled loss
param_grid = {
    'lr': [0.0001, 0.00001],
    'niter': [10, 20, 30],
    'niter_decay': [10, 20, 30],
    'criterion': ['input'],
    'threshold': [0.1],
    'weight_decay': [0.01, 0.1, 0.3, 0.5],
    'target': ['Ki67'],
    'lambda_u': [0.0, 0.1, 0.3, 0.5],
    'batchSize': [4, 8, 16]
    }


allNames = sorted(param_grid)
combinations = it.product(*(param_grid[Name] for Name in allNames))

for comb in combinations:

    # generate a unique name for the output folder
    for a, b in zip(allNames, comb):
        foldername = ''.join([str(a) + ':' + str(b) + '_'])

    if not os.path.exists(os.path.join('./checkpoints', foldername)):

        # generate the command for calling this run in python terminal
        command = "python train.py --dataroot datasets --dataset_mode labeled "
        command += "--which_model vgg_sara_small --roisize 16 "
        command += "--name " + foldername + ' '
        command += "--rotate "

        for a, b in zip(allNames, comb):
            command += " --%s %s " % (a, b)

        print(command)
        try:
            os.system(command)

        except Exception:
            err = 'exception' + Exception + ' raised in command :' + command
            print(err)

    else:
        print('folder exists: %s. skip.' % foldername)
