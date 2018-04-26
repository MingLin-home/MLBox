'''
The top-level user API for Machine Learning Pipeline.
Common option strings:
'--dataset_interface' : dataset python class file name to load
'--estimator_interface' : estimator python class file name to load
'--evaluation_interface' : evaluation python class file name to load
'''

import importlib
from argparse import ArgumentParser
import itertools
import sklearn.model_selection
import random
import os
import distutils.dir_util
import numpy as np
from MLBox import GPUMultiProcess


def gen_cross_validation_jobs(parameter_grid, option_str, cache_file_name=None):
    '''
    Generate cross-validation jobs. Output strings for job configuration as:
    option_str + ' --dataset_interface --dataset_name=%s --repeat_id=%d --cv_id=%d --parameter_name_1=%(g or %d or string) --parameter_name_2=%(g or %d or string)
    :dataset_interface: The dataset class python file name
    :parameter_grid: parameter grid of estimators to be tuned. Format: parameter_grid={'parameter_name_1':[0.01,0.1,1,10],'parameter_name_2':['a', 'b', 'c']}
    :option_str: any optional string to pass to the remaining pipeline. For example, the estimator class file, the evaluation class file, etc.
    :cache_file_name: if not None, cache the job list in cache_file_name
    :return: List, each element is a string specifying the job configuration
    '''
    if cache_file_name is not None and os.path.isfile(cache_file_name):
        fid = open(cache_file_name, 'r')
        jobs_string_list = fid.readlines()
        fid.close()
        return jobs_string_list
    pass

    option_str_split_list = option_str.split(' ')
    parser = ArgumentParser()
    # parser.add_argument('--dataset_interface', type=str, help='dataset python class file name to load')
    parser.add_argument('--dataset_name_list', type=str, help='dataset name list to load')
    parser.add_argument('--cv_method', type=str, default='rn',
                        help='cross-validation method, could be: rn (random, default), gs(grid search), cs (coordinate search). ')
    parser.add_argument('--random_cv_folds', type=int, default=0.1,
                        help='how many random cross-validation folds to perform, if integer, do so many searches. If float (<1), perform the persentage number of searches. Default:0.1')
    parser.add_argument('--repeat_id_list', type=str, help='repeat id list, can be int or str')
    parser.add_argument('--cv_id_list', type=str, help='cross-validation id list, can be int or str')
    (command_line_options, command_line_unknown_args_str) = parser.parse_known_args(option_str_split_list)

    # dataset_pyclass_mod = importlib.__import__(command_line_options.dataset_interface)
    # the_dataset = dataset_pyclass_mod.Dataset(option_str)

    dataset_name_list = command_line_options.dataset_name_list.split(',')
    repeat_id_list = command_line_options.repeat_id_list.split(',')
    cv_id_list = command_line_options.cv_id_list.split(',')

    jobs_string_list = []
    for dataset_name, repeat_id, cv_id in itertools.product(dataset_name_list, repeat_id_list, cv_id_list):
        the_param_grid = sklearn.model_selection.ParameterGrid(parameter_grid)
        tmp_job_list = []
        for the_param_dict in the_param_grid:
            the_job_str = option_str
            the_job_str += ' --{0} --repeat_id={1} --cv_id={2}'.format(dataset_name, repeat_id, cv_id)
            for param_name in the_param_dict.keys():
                the_param_value = the_param_dict[param_name]
                the_job_str += ' --{0}={1}'.format(param_name, the_param_value)
            pass # end for param_name
            tmp_job_list.append(the_job_str)
        pass # end for
        if command_line_options.cv_method == 'rn':
            actual_random_cv_folds = 0
            if command_line_options.random_cv_folds <= 1:
                actual_random_cv_folds = int(len(tmp_job_list) * command_line_options.random_cv_folds)
            else:
                actual_random_cv_folds = command_line_options.random_cv_folds
            pass
            random.shuffle(tmp_job_list)
            tmp_job_list = tmp_job_list[0:actual_random_cv_folds]
        pass

        jobs_string_list += tmp_job_list

    pass # end for dataset_name

    if cache_file_name is not None:
        fid = open(cache_file_name, 'w')
        fid.writelines([the_line + '\n' for the_line in jobs_string_list])
        fid.close()
    pass
    return jobs_string_list
pass

def par_do_train_eval(device_config_str_list, option_str):
    GPUMultiProcess.restric_computation_resource(device_config_str_list)
    train_eval(option_str)
pass
def train_eval(option_str):
    '''
    Train and Evaluate estimator on given dataset, repeat, cross-validation fold.
    '--num_cpus_per_job': number of CPUs to run each job
    '--output_dir': output directory
    :param option_str:
    :return:
    '''
    
    option_str_split_list = option_str.split(' ')
    parser = ArgumentParser()
    # parser.add_argument('--dataset_interface', type=str, help='dataset python class file name to load')
    parser.add_argument('--estimator_interface', type=str, help='estimator python class file name to load')
    # parser.add_argument('--evaluation_interface', type=str, help='evaluation python class file name to load')
    parser.add_argument('--num_cpus_per_job', type=str, help='number of CPUs to run each job')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--dataset_name', type=str, help='dataset name to load')
    parser.add_argument('--repeat_id', type=int, help='repeat experiemnt ID')
    parser.add_argument('--cv_id', type=int, help='cross-validation fold ID')
    # parser.add_argument('--save_estimator', action='store_true', help='save estimator if specified')
    parser.add_argument('--user_func_file', type=str, help='user-defined function python file name')
    parser.add_argument('--user_func_name', type=str, help='user-defined function name')

    (command_line_options, command_line_unknown_args_str) = parser.parse_known_args(option_str_split_list)

    # dataset_interface_mod = importlib.__import__(command_line_options.dataset_interface)
    # the_dataset_interface = dataset_interface_mod.Dataset(option_str)
    estimator_interface_mod = importlib.__import__(command_line_options.estimator_interface)
    the_estimator_interface = estimator_interface_mod.Estimator(option_str)
    # evaluation_interface_mod = importlib.__import__(command_line_options.evaluation_interface)
    # the_evaluation_interface = evaluation_interface_mod.Evaluation(option_str)
    user_func_mod = importlib.__import__(command_line_options.user_func_file)
    user_func = getattr(user_func_mod, command_line_options.user_func_name)

    output_dir_base = os.path.join(command_line_options.output_dir), 'On{}CPU/{}/{}/rp{}cv{}/'.format(
        command_line_options.num_cpus_per_job,
        command_line_options.dataset_name,
        the_estimator_interface.get_name(),
        command_line_options.repeat_id,
        command_line_options.cv_id,
    )
    distutils.dir_util.mkpath(output_dir_base)
    # trainset_evaluation_filename = os.path.join(output_dir_base, 'eval_trainset')
    # valset_evaluation_filename = os.path.join(output_dir_base, 'eval_valset')
    # testset_evaluation_filename = os.path.join(output_dir_base, 'eval_testset')

    # save_estimator_file_name = os.path.join(output_dir_base, 'saved_estimator')

#     must_train_estimator = False
#     if command_line_options.save_estimator and not os.path.isfile(save_estimator_file_name):
#         must_train_estimator = True
#     if not (os.path.isfile(trainset_evaluation_filename) and os.path.isfile(valset_evaluation_filename) and os.path.isfile(testset_evaluation_filename)):
#         must_train_estimator = True

#     if must_train_estimator:
#         if os.path.isfile(save_estimator_file_name):
#             the_estimator_interface.load(save_estimator_file_name)
#         else:
#             train_X, train_Y = the_dataset_interface.load_dataset(set_name='train')
#             the_estimator_interface.fit(train_X, train_Y)
#             if command_line_options.save_estimator:
#                 the_estimator_interface.save(save_estimator_file_name)
#             pass  # end if
#         pass # end if
#     pass # end if must_train_estimator

#     pred_train_Y = the_estimator_interface.predict(train_X)
#     the_evaluation_interface.evaluate(Y_true=train_Y, Y_pred=pred_train_Y, output_file_name=trainset_evaluation_filename)

#     val_X, val_Y = the_dataset_interface.load_dataset(set_name='val')
#     if val_X is not None:
#         pred_val_Y = the_estimator_interface.predict(val_X)
#         the_evaluation_interface.evaluate(Y_true=val_Y, Y_pred=pred_val_Y, output_file_name=valset_evaluation_filename)
#     else:
#         np.savetxt(valset_evaluation_filename, option_str)
#     pass # end if


#     test_X, test_Y = the_dataset_interface.load_dataset(set_name='test')
#     if test_X is not None:
#         pred_test_Y = the_estimator_interface.predict(test_X)
#         the_evaluation_interface.evaluate(Y_true=test_Y, Y_pred=pred_test_Y, output_file_name=testset_evaluation_filename)
#     else:
#         np.savetxt(testset_evaluation_filename, option_str)
#     pass # end if

    user_func(option_str=option_str, output_dir_base=output_dir_base, the_estimator_interface=the_estimator_interface)

pass

def parallel_train_eval(num_jobs, num_cpus_per_job, task_list):
    '''
    Parallelly train and evaluate. Option strings:
    :param option_str: the arguement string. To get parameter tuning configuration, call gen_cross_validation_jobs()
    :return:
    '''

    addition_option_str = ' --num_cpus_per_job={}'.format(num_jobs, num_cpus_per_job)

    GPUMultiProcess.parallel_GPU_CPU_do(
        num_gpus=0, num_cpus=num_jobs*num_cpus_per_job, par_do_func=par_do_train_eval,
        task_list=[i + addition_option_str for i in task_list], per_task_num_cpus=num_cpus_per_job
    )
    pass
pass
