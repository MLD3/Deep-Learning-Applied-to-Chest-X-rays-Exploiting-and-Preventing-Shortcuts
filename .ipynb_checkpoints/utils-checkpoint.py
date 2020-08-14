"""
EECS 445 - Introduction to Machine Learning
Fall 2018 - Project 2
Utility functions
"""
import os
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import metrics
import trainer
from sklearn.utils import resample  
import scipy 
from matplotlib import pyplot as plt

def config(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(config, 'config'):
        with open('config.json') as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split('.'):
        node = node[part]
    return node

def denormalize_image(image):
    """ Rescale the image's color space from (min, max)
    to (0, 1) """
    ptp = np.max(image, axis=(0,1)) - np.min(image, axis=(0,1))
    return (image - np.min(image, axis=(0,1))) / ptp

def hold_training_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.ioff()
    plt.show()

def log_cnn_training(epoch, stats):
    """
    Logs the validation accuracy and loss to the terminal
    """
    
    valid_auc, valid_loss, val_acc, train_auc, train_loss, train_acc = stats[-1]
    print('Epoch {}'.format(epoch))
    print('\tValidation Loss: {}'.format(valid_loss))
    print('\tValidation AUC: {}'.format(valid_auc))
    print('\tValidation Accuracy: {}'.format(val_acc))
    print('\tTrain Loss: {}'.format(train_loss))
    print('\tTrain AUC: {}'.format(train_auc))
    print('\tTrain Accuracy: {}'.format(train_acc))

def make_precision_recall(save_directory, y_true, y_pred, config_str, params, n_classes, labels, plot_labels = False, plot_location = None, CI = True):
    
    
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                            y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
        y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_pred,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))


    from itertools import cycle
    # setup plot details
    colors = ['r', 'b', 'g']

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    
    for i, color in zip(range(n_classes), colors):
        aupr_scores = []
        for j in range(1000):
            yte_true_b, yte_pred_b = resample(y_true[:,i], y_pred[:,i], replace=True, random_state=j)
            aupr_scores.append(metrics.auc(*metrics.precision_recall_curve(yte_true_b, yte_pred_b)[1::-1]))
    
        conf_int = ' ({:.2f}-{:.2f})'.format(np.percentile(aupr_scores, 2.5), np.percentile(aupr_scores, 97.5))
        test = 'Precision-recall for class {0} area = {1:0.2f}'.format(i, average_precision[i]) + conf_int
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append(test)

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
#     plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


    if (plot_location == None):
        filename = save_directory + train_common.create_checkpoint_string(params) +'precision_recall_plot.pdf'
    else:
        filename = plot_location + "_precision_recall_plot.pdf"
    print("Saving ROC curve at:", filename)
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()

def get_roc_CI(y_true, y_score):
#     roc_curves, auc_scores = zip(*Parallel(n_jobs=4)(delayed(bootstrap_func)(i, y_true, y_score) for i in range(1000)))
    roc_curves, auc_scores, aupr_scores = [], [], []
    for j in range(1000):
        try:
            yte_true_b, yte_pred_b = resample(y_true, y_score, replace=True, random_state=j)
            roc_curve = metrics.roc_curve(yte_true_b, yte_pred_b)
            auc_score = metrics.roc_auc_score(yte_true_b, yte_pred_b)
            aupr_score = metrics.auc(*metrics.precision_recall_curve(yte_true_b, yte_pred_b)[1::-1])
            
            roc_curves.append(roc_curve)
            auc_scores.append(auc_score)
            aupr_scores.append(aupr_score)
        except: 
            j -= 1
#     print('Test AUC: {:.3f}'.format(metrics.roc_auc_score(y_true, y_score)))
#     print('Test AUC: ({:.3f}, {:.3f}) percentile 95% CI'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))) 
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fpr, tpr, _ in roc_curves:
#         print(scipy.interp(mean_fpr, fpr, tpr))
        tprs.append(scipy.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(metrics.auc(fpr, tpr))
            
    mean_tpr = np.mean(tprs, axis=0)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
    return roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper

def make_auc_plot(save_directory, y_true, y_pred, config_str, params, n_classes, plot_labels = False, plot_location = None, CI = True):
    
    print("making plot")
    print("n_classes", n_classes)
    plt.tight_layout()

    colors = ['r','b','g']
    lw = 2
    fpr = []
    tpr = []
    auc = []
    labels = config(config_str + ".labels").split("|")
    if (plot_labels):
        plot_labels = plot_labels.split("|")
    else:
        plot_labels = config(config_str + ".graph_labels").split("|")
    if(n_classes == 1):
        color = colors[0]
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)

        
        auc = metrics.roc_auc_score(y_true, y_pred)
        
        roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper = get_roc_CI(y_true, y_pred)
        
        if (CI):
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.1, color=color)
            conf_int = ' ({:.2f}-{:.2f})'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
            test = 'ROC curve of ' + plot_labels[0] + ' area = {1:0.2f}'.format(0, auc) + conf_int
        else:
            test = 'ROC curve of ' + plot_labels[0] + ' area = {1:0.2f}'.format(0, auc)
        
        plt.plot(fpr, tpr, color=color, lw=lw,
                 label=test)
            
            
    else:
        for ind in range(len(labels)):
            
            f, t, _ = metrics.roc_curve(np.squeeze(y_true)[:,ind], np.squeeze(y_pred)[:,ind])
            fpr.append(f)
            tpr.append(t) 

            auc.append(metrics.roc_auc_score(np.squeeze(y_true)[:,ind], np.squeeze(y_pred)[:,ind]))
    
        for i, color in zip(range(len(plot_labels)), colors):
            roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper = get_roc_CI(np.squeeze(y_true)[:,i], np.squeeze(y_pred)[:,i])

            if (CI):
                plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.1, color=color)
                conf_int = ' ({:.2f}-{:.2f})'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
                test = 'ROC curve of ' + plot_labels[i] + ' area = {1:0.2f}'.format(i, auc[i]) + conf_int
            else:
                test = 'ROC curve of ' + plot_labels[i] + ' area = {1:0.2f}'.format(i, auc[i])
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=test)
                 
    plt.axis('scaled')
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    if (plot_location == None):
        filename = save_directory + train_common.create_checkpoint_string(params) +'cnn_roc_plot.pdf'
    else:
        filename = plot_location + "_roc_plot.pdf"
    print("Saving ROC curve at:", filename)
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()

    return 
def make_cnn_training_plot(config_str, params):
    """
    Runs the setup for an interactive matplotlib graph that logs the loss and
    accuracy
    """
    fig, axes = plt.subplots(1,1, figsize=(10,5))
    plt.suptitle(config(config_str+ ".plot_title") + ' Training')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    param_str = "Parameters: " + train_common.create_checkpoint_string(params)
    plt.text(.5, .05, param_str, ha='center')
    return axes 

def update_cnn_training_plot(save_directory, config_str, epoch, stats, params):
    """
    Updates the training plot with a new data point for loss and accuracy
    """
    lr = params["lr"]
    batch_size = params["batch_size"]
    axes = make_cnn_training_plot(config_str, params) 
    valid_loss = [s[1] for s in stats]
    train_loss = [s[4] for s in stats]
    axes.plot(range(epoch - len(stats) + 1, epoch + 1), valid_loss,
        linestyle='--', marker='o', color='b')
    axes.plot(range(epoch - len(stats) + 1, epoch + 1), train_loss,
        linestyle='--', marker='o', color='r')
    axes.legend(['Validation', 'Train'])
    param_str = "Parameters: " + train_common.create_checkpoint_string(params)
    plt.annotate(param_str, (0,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=14)
    filename = save_directory
    print("save directory:", save_directory)
    for key, value in params.items(): 
        filename += key + "_" + str(value) + "_"
    filename += "cnn_training_plot.png"
    print("Saving training plot at at:", filename)
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_auc(y_true, y_pred, labels):
    plt.tight_layout()

    colors = ['r','b','g', 'c', 'm']
    lw = 2
    fpr = []
    tpr = []
    auc = []
    n_classes = len(labels)
        
    if(n_classes == 1):
        color = colors[0]
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)

        
        auc = metrics.roc_auc_score(y_true, y_pred)
        
        roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper = get_roc_CI(y_true, y_pred)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.1, color=color)
        conf_int = ' ({:.2f}-{:.2f})'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
        test = 'ROC curve of ' + labels[0] + ' area = {1:0.2f}'.format(0, auc) + conf_int
        print(test)
        plt.plot(fpr, tpr, color=color, lw=lw,
                 label=test)            
    else:
        for ind in range(len(labels)):
            f, t, _ = metrics.roc_curve(np.squeeze(y_true)[:,ind], np.squeeze(y_pred)[:,ind])
            fpr.append(f)
            tpr.append(t) 
            auc.append(metrics.roc_auc_score(y_true[:,ind], y_pred[:,ind]))
    
        for i, color in zip(range(len(labels)), colors):
            roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper = get_roc_CI(np.squeeze(y_true)[:,i], np.squeeze(y_pred)[:,i])

            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.1, color=color)
            conf_int = ' ({:.2f}-{:.2f})'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
            if i == 0:
                test = 'ROC curve of ' + labels[i] + ' area = {1:0.2f}'.format(i, auc[i]) + conf_int
            elif i == 1:
                test = 'ROC curve of ' + labels[i] + ' area = {1:0.2f}'.format(i, auc[i]) + conf_int
            else:
                test = 'ROC curve of ' + labels[i] + ' area = {1:0.2f}'.format(i, auc[i]) + conf_int

#             else:
#                 test = 'ROC curve of ' + labels[i] + ' area = {1:0.2f}'.format(i, auc[i])
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=test)
            print(test)
    plt.axis('scaled')
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', fontsize = 'small')
    plt.savefig("fused_auroc.png", dpi = 1000)
    plt.show()
    return auc[i], np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5)
    
    

def get_auc(y_true, y_pred, labels):
    fpr = []
    tpr = []
    auc = []
    n_classes = len(labels)
        
    if(n_classes == 1):
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)

        
        auc = metrics.roc_auc_score(y_true, y_pred)
        
        roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper = get_roc_CI(y_true, y_pred)
        conf_int = ' ({:.2f}-{:.2f})'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
        return '{1:0.2f}'.format(0, auc) + conf_int
    else:
        strings = []
        for ind in range(len(labels)):
            f, t, _ = metrics.roc_curve(np.squeeze(y_true)[:,ind], np.squeeze(y_pred)[:,ind])
            fpr.append(f)
            tpr.append(t) 
            auc.append(metrics.roc_auc_score(y_true[:,ind], y_pred[:,ind]))
    
        for i in range(len(labels)):
            roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper = get_roc_CI(np.squeeze(y_true)[:,i], np.squeeze(y_pred)[:,i])

            conf_int = ' ({:.2f}-{:.2f})'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
            strings.append('{1:0.2f}'.format(i, auc[i]) + conf_int)
        return strings 
    
def plot_average_auc(predictions, labels, colors):
    
    for idx, label in enumerate(labels):
        roc_curves = []
        auc_scores = []
        for [y_true, y_pred] in predictions:
            roc_curves.append(metrics.roc_curve(y_true[:, idx], y_pred[:, idx])) 
            auc_scores.append(metrics.roc_auc_score(y_true[:, idx], y_pred[:, idx]))

        print(auc_scores)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for fpr, tpr, _ in roc_curves:
            tprs.append(scipy.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(metrics.auc(fpr, tpr))


        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        legend = 'ROC curve of ' + label + ' area = {1:0.2f}'.format(idx, np.mean(auc_scores))

        plt.plot(mean_fpr, mean_tpr, lw = 1.25, label = legend, color = colors[idx])
    plt.axis('scaled')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    