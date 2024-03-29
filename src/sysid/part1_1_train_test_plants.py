"""Sample code.

J R Forbes, 2022/03/04

This code loads the data.
"""

# Libraries
from matplotlib import pyplot as plt
import pathlib
import plant_util as util
import time
import numpy as np

def parallel_train_test_process(process_material: util.PlantProcessMaterial) -> util.PlantDesignOutcome:

    s_data_train, s_data_test = process_material.load_train_test_data()
    design_outcome = util.PlantDesignOutcome(process_material.name)
    design_outcome.id_plant(s_data_train, process_material.design_params)

    # train_perform = design_outcome.train_perform

    # print(train_perform.satisfying_attributes)
    # print(train_perform.conditioning)
    # print(train_perform.relative_uncertainty)

    # print(f'Good plant? :{train_perform.is_completly_satisfying}')
    #if train_perform.is_completly_satisfying:
    
    graph_data = design_outcome.test_plant(s_data_test, process_material.graph_data_cmds)

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

    # Plotting parameters

    plt.rc('lines', linewidth=2)
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle='--')


    # %%
    # Time to train (and test) plants

    start_t = time.perf_counter()
    
    try_all = False
    
    MAIN_FILE_PATH = util.find_io_files_folder()

    DATA_PATH = MAIN_FILE_PATH / 'DATA/'

    TRAINING_PLAN_PATH = MAIN_FILE_PATH / 'training_plans/plan.json'

    VERSION = 'v2'

    if try_all:
        # Here we train over a range of parameters, determined by 
        # build_regularization_array and the 'plants_to_train.json' file

        graph_data_cmds = util.PlantGraphDataCommands()
        # Build array of possible regularization values 
        regs = util.build_regularization_array(-2, 0, 1.1, 30)
        # Let's build a list of different training/testing scenarios to be executed on different processes
        pmg = util.PlantPMGTryAll(DATA_PATH, graph_data_cmds, TRAINING_PLAN_PATH, regs)
    else:
        # Here we train only using certain parameters we have found to be optimal 
        # when looking at results obtained when using PlantPMGTryAll
        # This is done for example to visualize some underlying computations, but only for the
        # plants that best recreate data found in DATA_PATH

        # Change save_graph to True to better visualize if optimal trained plants can recreate output signals from input
        graph_data_cmds = util.PlantGraphDataCommands(save_graph=True, show=False)

        # Select file in which to save these optimal parameters
        result_names_file_path = MAIN_FILE_PATH / f'good_plants/{VERSION}/good_plants_{VERSION}.txt'
        # Let's recreate a list of good training/testing scenarios to be executed (again) on different processes
        pmg = util.PlantPMGTrySelection(DATA_PATH, graph_data_cmds, TRAINING_PLAN_PATH, result_names_file_path)

    design_results = util.test_all_possible_trained_plants(pmg, parallel_train_test_process)

    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f'Total Processing took {total_duration:.2f} seconds')

    # Put all trained plants in a file, with test results
    if try_all:
        # Save a file with trained plants and corresponding statistics
        design_results.save_results_to_file(MAIN_FILE_PATH / f'try_all_results/results_plants_trained.json')
        # Only save the trained plant coefficients: this is the only thing we need going forward
        design_results.save_plant_list_to_file(MAIN_FILE_PATH / f'try_all_results/plants_trained.json')
    else:
        design_results.save_results_to_file(MAIN_FILE_PATH / f'good_plants/{VERSION}/results_plants_trained_{VERSION}.json')
        design_results.save_plant_list_to_file(MAIN_FILE_PATH / f'good_plants/{VERSION}/plants_trained_{VERSION}.json')
    # %%
    # Show plots
    # plt.show()