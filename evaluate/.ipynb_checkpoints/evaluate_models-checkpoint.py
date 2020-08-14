import torch
import numpy as np
from experiment import Experiment 
from trainer import predictions 
import pandas as pd 
from sklearn import metrics
import torch.nn as nn


def get_best_model_across_seeds(df_search):
    average_scores = []
    for score in df_search['best_score']:
        average_scores.append(np.mean(eval(score)))
    df_search["average_score"] = average_scores


    #     get sets of hyperparameters used 
    seeds = df_search.seed.unique()

    params = pd.DataFrame()

    for idx, row in df_search.iterrows():
        params = params.append(row.drop(['savename', 'best_iter', 'seed','average_score', 'best_score']).to_dict(), ignore_index = True)
    #  
    params = params.drop_duplicates()
    # index by number of seeds 
    best_average_performance = 0 
    best_params = {}
    best_index= 0
    not_done = False
    for i in range(0, len(df_search), len(seeds)):
        try:
            total_perf = 0
            for j in range(len(seeds)):
                total_perf += df_search.average_score[i + j]
            total_perf /= len(seeds)
            if (total_perf > best_average_performance):
                best_average_performance = total_perf 
                best_params = df_search.iloc[i].drop(['savename', 'best_iter', 'seed','average_score', 'best_score']).to_dict()
                best_index = i 
        except:
            not_done = True 
            print("Not done training")
            break
    best_model_info = []
    for k in range(len(seeds)):
        best_model_info.append(df_search.iloc[best_index + k])
    
    return best_model_info, not_done 

def get_best_model_info(df_search):
    average_scores = []
    for score in df_search['best_score']:
        average_scores.append(np.mean(eval(score)))
    df_search["average_score"] = average_scores
    
    print(df_search.columns)
    
#     get sets of hyperparameters used 
    df_search_sorted = df_search.sort_values('average_score', ascending=False).head()
    
    
    for idx, row in df_search_sorted.iterrows():
        params = row.drop(['savename', 'best_iter', 'seed','average_score', 'best_score']).to_dict()

    best_model_info = df_search_sorted.iloc[0]
    return best_model_info

def load_best_model(best_model_info, device, config_str, model_type, model_name,  load_filename=None):
    if load_filename is None:
        savename = best_model_info['savename']
        split = savename.split('/')
        split[-1] = 'best_' + split[-1]
        load_filename = '/'.join(split)
    
    try:
        _iter = checkpoint['_iter']
    except:
        _iter = checkpoint['_epoch']
    print("Loaded checkpoint (trained for {} iterations)".format(_iter))
    
    try:
        params = best_model_info[["momentum", "seed", "lr", "batch_size", "augmentation"]]
    except:
        params = best_model_info[["seed", "lr", "batch_size", "augmentation"]]

    params = params.to_dict()
    
    params_alt = {}
    
    for key, value in params.items():
        if (type(value) == str):
            params_alt[key] = [eval(value)]
        else:
            params_alt[key] = [value]

    exp = Experiment(None, device, config_str, model_type, model_name, params_alt, None, None)

    model, criterion, optimizer = exp._get_model(best_model_info['seed'], params)
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)
    
    return checkpoint, model, criterion,exp 

def get_test_predictions(model, criterion, device, te_loader, task=None, model_name=None, get_all_predictions = False):
    model.eval()
    running_pred = []
    running_pred_image = []
    running_pred_ehr = []
    pt_ids = []
    y_orig_all = []
    with torch.no_grad():
        for X, y, pt_id in te_loader:
            for p in pt_id:
                pt_ids.append(p)
            if (type(X) is list):
                for i in range(len(X)):
                    X[i] = X[i].to(device).half()
                    y = y.to(device).half()
            else:  
                X,y = X.to(device).half(), y.to(device).half()

#             print("model name:", model_name)
            if ("bias_" in model_name[0:5]):
                
                output = model(X) 
#                 print("prediction shape:", output[0].shape, output[1].shape)
                if (get_all_predictions):
                    predicted = (predictions(output[0].data), predictions(output[1].data))
                    running_pred_image.append((predicted[0].data.detach().cpu().numpy(), y.data.detach().cpu().numpy()))
                    running_pred_ehr.append((predicted[1].data.detach().cpu().numpy(), X[1].data.detach().cpu().numpy()))
                    y_pred_image, y_true_image = zip(*running_pred_image)
                    y_pred_ehr, y_true_ehr = zip(*running_pred_ehr)
                    y_pred = (y_pred_image, y_pred_ehr)
                    y_true = (y_true_image, y_true_ehr)

                else:
                    predicted = predictions(output[0].data)
                    running_pred.append((predicted.data.detach().cpu().numpy(), y.data.detach().cpu().numpy()))
                    y_pred, y_true = zip(*running_pred)

            else:
                output = torch.squeeze(model(X))
#                 print(output)
                y = torch.squeeze(y)

                #         mask loss 
#                 output[y<0] = y[y<0]

                predicted = predictions(output.data)

                running_pred.append((predicted.data.detach().cpu().numpy(), y.data.detach().cpu().numpy()))
                y_pred, y_true = zip(*running_pred)
    return y_true, y_pred, pt_ids
          
        

def save_test_predictions(checkpoint_str, pts, y_true, y_score,  model_name):
    import pathlib
    pathlib.Path('./output/{}/'.format(checkpoint_str)).mkdir(parents=True, exist_ok=True)
    
    fname = './output/{}/'.format(checkpoint_str)
          
    np.savez(
        open(fname, 'wb'),
        y_score = y_score,
        y_true  = y_true,
        pts = pts,
    )
          
    print('Test predictions saved to', fname)

    
def calc_roc(y_true, y_pred, pt_ids):
    

    unique_pt_ids = np.unique(pt_ids)
    unique_predictions = []
    unique_truth_values = []
    for pt_id in unique_pt_ids:
        indices = np.where(pt_ids == pt_id)[0]

        if(len(y_pred[indices]) == 1):
            unique_predictions.append(y_pred[indices][0])
            unique_truth_values.append(y_true[indices][0])
        else:
            unique_predictions.append(np.average(y_pred[indices], axis = 0))
            unique_truth_values.append(np.average(y_true[indices], axis = 0))
            
    unique_predictions = np.squeeze(np.array(unique_predictions))

    unique_truth_values = np.squeeze(np.array(unique_truth_values))

        
#         mask loss 
    unique_predictions[unique_truth_values<0] = unique_truth_values[unique_truth_values<0]
    if (len(y_pred.shape) == 1):
        y_pred = np.expand_dims(y_pred, axis = 1) 
    n_classes = y_pred.shape[1]
#     print("num classes:", n_classes)
    score = []

    if (n_classes == 1):
        score.append(metrics.roc_auc_score(unique_truth_values.astype(int), unique_predictions))
    else:
        for n in range(n_classes):
            try:
                score.append(metrics.roc_auc_score(unique_truth_values[:,n].astype(int), unique_predictions[:,n]))
            except:
                score.append(0.5)
    return score, unique_predictions, unique_truth_values, unique_pt_ids