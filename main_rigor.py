from helper.data_helper import *
from helper.training_helper import *
from helper.loss_helper import *
from helper.online_helper import *
from helper.plotting_helper import *


import argparse
import numpy as np
import json





def main(test_size=0.2, epsilon=.1, C=16, K=1, para_id='1', gamma='auto', dir=None):
    # Generate data for training
    
    print('epsilon: ', epsilon)
    print('C: ', C)
    print('K: ', K)
    print('test_size', test_size)
    print('gamma', gamma)

    print('Loading data...')

    # u, s, sf, cw, solver = data_loader(state=state)
    u, s, sf, cw = control_data_loader()

    print('Preprocessing')
    U_train, U_test, s_train, s_test, sc_U, sc_S = control_dataset(u, s, sf, cw, K=K, test_size=test_size)

    print('1d Trial')
    report_1d = svr_training_1d(U_train, U_test, s_train, s_test, epsilon, C, gamma)

    print('Real Training')
    model, report_multidim = svr_training_multidim(U_train, U_test, s_train, s_test, epsilon, C, gamma)
    
    n_support_array = np.zeros(len(model.estimators_))
    
    for i in range(len(model.estimators_)):
        n_support_array[i] = model.estimators_[i].n_support_

    mean_n_support = float(n_support_array.mean())
    
    
    save_model(model, 'model_'+para_id)
    save_model(sc_U, 'sc_U_'+para_id)
    save_model(sc_S, 'sc_S_'+para_id)
    
    save_np_array(U_train, 'U_train_'+para_id)

    initial_condition = None

    print('Online Testing')

    control_sf_cw_para = {
        'sfe':3.6e-3, 
        'sfv':9e-8, 
        'cwe':32, 
        'cwv':256, 
        'corr':0.75, 
        'seed':21*6+7+1
    }

    solver_ML, u_ML, duration = fast_online_testing(initial_condition, model, U_train, sc_U, sc_S, K, gamma, t_max=360*96*86400, w=3e-4, state=control_sf_cw_para)
    
    save_np_array(u_ML, 'u_ML_'+para_id)

    qbo_paras = para_for_plotting(solver_ML, u_ML)


    report =  {
            'epsilon': epsilon,
            'C': C,
            'K': K,
            'test_size': test_size,
            'gamma': gamma
        }

    report.update(report_1d)
    report.update(report_multidim)
    report.update(qbo_paras)
    report['qbo_objective'] = qbo_objective(**qbo_paras)
    report['online_duration'] = duration
    report['mean_n_support'] = mean_n_support

    print(report)
    

    with open('report.json', 'w') as f:
        json.dump(report, f)

    # dir = dir.strip()
    # if dir[-1] == '/':
    #     path = dir + 'all_report.json'
    # else:
    #     path = dir +'/all_report.json'


    # with open(path, 'r') as f:
    #     file_data = json.load(f)
    
    # file_data[para_id] = report
    
    # with open(path, 'w') as f:
    #     json.dump(file_data, f)
    
    if gamma == 'auto':
        gamma = 1 / 75
    
    text = f'test_size = {test_size:.3f}, epsilon = {epsilon:.5f}, C = {int(C):2d}, K = {K:.2f}, gamma = {gamma:.5f}'


    plot_76_tensors(u_ML, solver_ML, isu=True, text=text, file_prefix=para_id, **qbo_paras)

    # plot_wind_level(u_ML, level=35, text=text, file_prefix=para_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('-C', '--C',type=float, help="Speciy hyperparameter C")
    # parser.add_argument('-e', '--epsilon', type=float, help="Speciy hyperparameter epsilon")
    # parser.add_argument('-K', '--K', type=float, help="Speciy hyperparameter K")

    parser.add_argument('-p', '--parameter', help='Specify the parameter_id')
    parser.add_argument('-d', '--directory', help='base-directory')

    args = parser.parse_args()

    # epsilon, C, K = args.epsilon, args.C, args.K
    para_id = args.parameter
    dir = args.directory

    STATE = 1

    paras = load_parameters(dir, para_id)

    main(para_id=para_id, dir=dir, **paras)