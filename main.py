from SVR_helper import *
import argparse
import numpy as np
import json

def main(state=1, epsilon=.1, C=16, K=1, para_id='1', dir=None):
    print('epsilon: ', epsilon)
    print('C: ', C)
    print('K: ', K)

    print('Loading data...')
    u, s, sf, cw, solver = data_loader(state=state)

    print('Preprocessing')
    U_train, U_test, s_train, s_test, sc_U, sc_S = dataset(u, s, sf, cw, K=K)

    print('1d Trial')
    report_1d = training_1d(U_train, U_test, s_train, s_test, epsilon, C)

    print('Real Training')
    model, report_multidim = training_multidim(U_train, U_test, s_train, s_test, epsilon, C)

    save_model(model, 'model_'+para_id)
    save_model(sc_U, 'sc_U_'+para_id)
    save_model(sc_S, 'sc_S_'+para_id)

    initial_condition = lambda z: u[-1, :] 

    print('Online Testing')
    solver_ML, u_ML, duration = online_testing(initial_condition, model, sc_U, sc_S, K, state=state)

    qbo_paras = para_for_plotting(solver_ML, u_ML)


    report =  {
            'epsilon': epsilon,
            'C': C,
            'K': K,
        }

    report.update(report_1d)
    report.update(report_multidim)
    report.update(qbo_paras)
    report['qbo_objective'] = qbo_objective(**qbo_paras)
    report['online_duration'] = duration

    print(report)
    

    with open('report.json', 'w') as f:
        json.dump(report, f)



    dir = dir.strip()
    if dir[-1] == '/':
        path = dir + 'all_report.json'
    else:
        path = dir +'/all_report.json'


    with open(path, 'r') as f:
        file_data = json.load(f)
    
    file_data[para_id] = report
    
    with open(path, 'w') as f:
        json.dump(file_data, f)
    
    text = f'epsilon = {epsilon:.4f}, C = {int(C):2d}, K = {K:.2f}'


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

    main(state=STATE, para_id=para_id, dir=dir, **paras)