"""Sample code.

J R Forbes, 2022/03/04

This code loads the data.
"""

# Libraries
import numpy as np
# from scipy import signal
from matplotlib import pyplot as plt
import pathlib
import part_1_util as util


import time

# TODO: put process material in a class 
def parallel_train_test_process(process_material: tuple[util.PlantPreGeneratedDesignParams, int, pathlib.Path, float, util.PlantGraphDataCommands]) -> util.PlantDesignOutcome:
    pregen_params, i, regularization, graph_data_cmds = process_material
    params = util.complete_design_params(pregen_params, regularization)
    s_data_train, s_data_test = util.load_train_test_data(i)
    name = f'plant_n{params.num_order}_d{params.denum_order}_{params.better_cond_method[0:2]}_ds{i}_reg{regularization}'
    design_outcome = util.PlantDesignOutcome(name)
    design_outcome.regularization = regularization
    design_outcome.id_plant(s_data_train, params)

    train_perform = design_outcome.train_perform

    # print(train_perform.satisfying_attributes)
    # print(train_perform.conditioning)
    # print(train_perform.relative_uncertainty)

    # print(f'Good plant? :{train_perform.is_completly_satisfying}')
    #if train_perform.is_completly_satisfying:
    
    graph_data = design_outcome.test_plant(s_data_test, graph_data_cmds)

    # test_perform = design_outcome.test_perform
    print(f'{design_outcome.name} completed.')
    return design_outcome
        # print(test_perform.satisfying_attributes)
        # print(test_perform.testing_satisfying_attributes)
        # print(name)
        # print(f'cond    = {train_perform.conditioning}')
        # print(f'rel_unc = {train_perform.relative_uncertainty}')
        # print(f'VAF     = {test_perform.VAF}')
        # print(f'FIT     = {test_perform.FIT}')

        # print(f'Good Good plant? :{test_perform.is_completly_satisfying}')
    # else:
    #     return None

if __name__ == '__main__':
    # params 

    debug_print = True
    reduce_interm_graph = False

   
    # Plotting parameters
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif', size=14)
    plt.rc('lines', linewidth=2)
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle='--')


    # %%
    # Time to trian plants

    start_t = time.perf_counter()
    
    try_all = False
    if try_all:
        # Here we train over a range of parameters, determined by 
        # build_regularization_array and the 'plants_to_train.json' file
        graph_data_cmds = util.PlantGraphDataCommands()
        # Build array of possible regularization values 
        regs = util.build_regularization_array(-1, 1, 1.1, 30)
        pmg = util.PlantPMGTryAll(graph_data_cmds, 'training_plans/plants_to_train.json', regs)
    else:
        # Here we train only using certain parameters we have found to be optimal before
        graph_data_cmds = util.PlantGraphDataCommands(save_graph=False)
        result_names_file_path = pathlib.Path('good_plants/v3/good_plants_v3.txt')
        pmg = util.PlantPMGTrySelection(graph_data_cmds, result_names_file_path)

    design_results = util.test_all_possible_trained_plants(pmg, parallel_train_test_process)

    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f'Total Processing took {total_duration:.2f}s total')

    
    # Put all trained plants in a file, with test results
    design_results.save_results_to_file('good_plants/v3/results_plants_trained_v3.json')
    design_results.save_plant_list_to_file('good_plants/v3/plants_trained_v3.json')
    # %%
    # Show plots
    # plt.show()