
import argparse
import sys
import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import shutil

motor_symptoms = ['FutureDyskynesia', 'FutureMotorFluctuations', 'FutureFreezing']
models = ['randomforestclassifier', 'extratreesclassifier', 'xgbclassifier', 'logisticregression', 'svc', 'voting', 'stacking']


def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not found.")
    except PermissionError:
        print(f"Permission denied to delete '{folder_path}'.")
    except Exception as e:
        print(f"Error deleting folder '{folder_path}': {e}")


def gett_2(motor_symptom, classifier, n_feat, main_path):
    # path of raw results
    raw_res = pathlib.Path(f'{main_path}/{motor_symptom}/f_{n_feat}/{classifier}/raw_results.csv')
    res_path = pathlib.Path(f'{main_path}/{motor_symptom}/f_{n_feat}/{classifier}/dist_plots')
    res_path.mkdir(exist_ok=True, parents=True)
    #delete_folder(f'../results_x_neuro_clusters/{motor_symptom}/{classifier}/active_patients_plots')

    # load raw results for the model -> more than one result per patient
    data = pd.read_csv(raw_res)
   

    plt.figure(figsize=(10, 6))
    sns.histplot(data['prob y=1'], bins=20, kde=True)
    plt.xlim(0, 1)  # Set x-axis range from 0 to 1
    plt.title(f'Histogram of Predicted Probabilities | {classifier}')
    plt.xlabel('Predicted Probability of y=1')
    plt.ylabel('Frequency')
    plt.savefig(res_path / 'hist_plot.png', dpi=400)
    plt.close()


    plt.figure(figsize=(10, 6))
    sns.histplot(data[data['real y'] == 1]['prob y=1'], bins=20, color='blue', label='True Positives', kde=True)
    plt.xlim(0, 1)  # Set x-axis range from 0 to 1
    sns.histplot(data[data['real y'] == 0]['prob y=1'], bins=20, color='red', label='True Negatives', kde=True, alpha=0.5)
    plt.xlim(0, 1)  # Set x-axis range from 0 to 1
    plt.title(f'Histogram of Predicted Probabilities by True Label | {classifier}')
    plt.xlabel('Predicted Probability of y=1')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(res_path / 'hist_plot2.png', dpi=400)
    plt.close()


    fpr, tpr, _ = roc_curve(data['real y'], data['prob y=1'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic | {classifier}')
    plt.legend(loc="lower right")
    plt.savefig(res_path / 'roc_curve.png', dpi=400)
    plt.close()


    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))


    # Plot for True Negatives (real y = 0)
    sns.histplot(data[data['real y'] == 0]['prob y=1'], bins=20, color='red', label='True Negatives', kde=True, alpha=0.5, ax=axes[0])
    plt.xlim(0, 1)  # Set x-axis range from 0 to 1
    axes[0].set_title(f'Histogram of True Negatives | {classifier}')
    axes[0].set_xlabel('Predicted Probability of y=1')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()


    # Plot for True Positives (real y = 1)
    sns.histplot(data[data['real y'] == 1]['prob y=1'], bins=20, color='blue', label='True Positives', kde=True, ax=axes[1])
    plt.xlim(0, 1)  # Set x-axis range from 0 to 1
    axes[1].set_title(f'Histogram of True Positives | {classifier}')
    axes[1].set_xlabel('Predicted Probability of y=1')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()



    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(res_path / 'True_pos_neg_patients.png', dpi=400)
    plt.close()


def get_ovrall_res(folder_path, n_feat):

    motor_symptoms = ['FutureDyskynesia']
    for motor_symptom in motor_symptoms:
        res_path = pathlib.Path(f"{folder_path}/{motor_symptom}/f_{n_feat}")
        mean_m_list = list()
        std_m_list = list()
        for classifier in models:
            # path of raw results
            mean_metrics = pathlib.Path(f'{folder_path}/{motor_symptom}/f_{n_feat}/{classifier}/mean_metrics.csv')
            # load raw results for the model -> more than one result per patient
            mean_metrics = pd.read_csv(mean_metrics).drop(columns=['Unnamed: 0'])
            mean_m_list.append(mean_metrics.iloc[[0]]) #get first row (mean)
            std_m_list.append(mean_metrics.iloc[[1]]) #get secoind row (std)
            
        

        mean_metrics_df = pd.concat([li for li in mean_m_list], axis=0)
        mean_metrics_df.set_index(np.array(models), inplace=True)
        mean_metrics_df.to_csv(res_path / 'mean_metrics.csv')

        std_metrics_df = pd.concat([li for li in std_m_list], axis=0)
        std_metrics_df.set_index(np.array(models), inplace=True)
        std_metrics_df.to_csv(res_path / 'std_metrics.csv')


def main(main_path) -> int:
    n_feats = range(1,21)
        
    for symptom in motor_symptoms:
        for model in models:
            for n_feat in n_feats:
                gett_2(symptom, model, n_feat, main_path)

    ''' Group mean metrics in the main folder for the specific motor symptom'''
    for n_feat in n_feats:
        get_ovrall_res(main_path, n_feat)

    return 0

'''  DISTRIBUTION OF PREDICTED PROBABILITIES 
get distplots for models ran on subset of patients '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run script for additional plots")
    parser.add_argument("--path", type=str, help="Path to the main folder", required=True)
    
    args = parser.parse_args()
    
    sys.exit(main(main_path=args.path))