from __future__ import print_function

from hyperopt_tuner import *
import json

from train import *

max_trial_num = 200
path = './search_space.json'

tuner = HyperoptTuner("tpe", optimize_mode='maximize')

with open(path) as f:
    search_space = json.load(f)

print("Search space\t", search_space)

tuner.update_search_space(search_space)

for i in range(max_trial_num):
    params = tuner.generate_parameters(i)

    print("Params\t", params)

    value = test(params)
    # print("Execute done.")

    tuner.receive_trial_result(i, params, value)
    print("Receive value\t", value)

    
