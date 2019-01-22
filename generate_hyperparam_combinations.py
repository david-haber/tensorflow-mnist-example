import itertools
import yaml


hyper_params = {
    'learning_rate': [0.00001, 0.001, 0.002, 0.004, 0.001, 0.05, 0.1],
}

keys, values = zip(*hyper_params.items())
hyperparam_allsets = [dict(hyperparam_set=dict(zip(keys, v))) for v in itertools.product(*values)]

print("Total number of hyperparameter sets: " + str(len(hyperparam_allsets)))

with open('hyperparams.yml', 'w') as outfile:
    yaml.dump(hyperparam_allsets, outfile, default_flow_style=False)

print("Hyperparameter sets saved to: " + 'hyperparams.yml')
