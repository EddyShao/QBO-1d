from SVR_helper import *
import argparse
import numpy as np
import json
import joblib


def main(state, dir, para_id):
    # ground truth
    u, s, sf, cw, solver = data_loader(state=state)
    initial_condition = lambda z: u[-1, :] 

    # Load model
    paras = load_parameters(dir, para_id)
    
    dir = dir.strip()
    if dir[-1] != '/':
        dir = dir + '/'

    model_dir = dir + 'model_' + para_id + '/'

    model = load_model(model_dir+'model_'+para_id+'.pkl')
    sc_U = load_model(model_dir+'sc_U_'+para_id+'.pkl')
    sc_S = load_model(model_dir+'sc_S_'+para_id+'.pkl')

    K = paras['K']
    C = paras['C']
    epsilon = paras['epsilon']

    # Offline
    nsteps = 360*108
    nspinup = 360*12
    U = u[nspinup:nsteps, :]
    SF = sf[nspinup:nsteps]
    CW = cw[nspinup:nsteps]
    S = s[nspinup:nsteps, :]
    U = torch.hstack([U[:, 1:-1], SF.view(-1, 1), CW.view(-1, 1)])

    S_test = sc_S.transform(S)
    U_test = sc_U.transform(U)
    U_test[:, -2:] = K * U_test[:, -2:]

    with joblib.parallel_backend(backend='threading', n_jobs=16):
        perturbed_r2 = model.score(U_test, S_test)
        prediction = model.predict(U_test)
        perturbed_rmse = rMSE(prediction, S_test, S_test.std()+1e-32)
    
    # Online
    solver_ML, u_ML, duration = online_testing(initial_condition, model, sc_U, sc_S, K, state=state)

    qbo_paras = para_for_plotting(solver_ML, u_ML)

    text = f'PERTURBED_STATE = {state} | epsilon = {epsilon:.4f}, C = {int(C):2d}, K = {K:.2f}'

    plot_76_tensors(u_ML, solver_ML, isu=True, text=text, file_prefix=str(state)+'_perturbed_'+para_id, **qbo_paras)

    perturb_report = {
        str(state)+'_perturbed_r2': perturbed_r2,
        str(state)+'_perturbed_mean_rMSE': perturbed_rmse.mean(),
        str(state)+'_perturbed_amp25': qbo_paras['amp25'],
        str(state)+'_perturbed_amp20': qbo_paras['amp20'],
        str(state)+'_perturbed_period': qbo_paras['period']
    }



    with open(model_dir+'report.json', 'r') as f:
        file_data = json.load(f)
    
    file_data.update(perturb_report)
    
    with open(model_dir+'report.json', 'w') as f:
        json.dump(file_data, f)



    all_report_path = dir + 'all_report.json'

    with open(all_report_path, 'r') as f:
        file_data = json.load(f)

    file_data[para_id].update(perturb_report)
    
    with open(all_report_path, 'w') as f:
        json.dump(file_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('-p', '--parameter', help='Specify the parameter_id')
    parser.add_argument('-d', '--directory', help='base-directory')
    parser.add_argument('-s', '--state', type=int, help='perturbation state')

    args = parser.parse_args()

    para_id = args.parameter
    dir = args.directory
    state=args.state

    main(state=state, para_id=para_id, dir=dir)





    


    


    



    

    


    


    

    

    






    