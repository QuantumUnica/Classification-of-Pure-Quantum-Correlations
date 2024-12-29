import pandas as pd
import numpy as np
import os
import glob
import csv
from collections import defaultdict
from itertools import combinations, product
import utils

import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 150

def get_classifier_vs_dataset_heatmap(score_dFrame=None, metric='balanced_accuracy', save_path_img=None):     
    
    classifiers = score_dFrame['classifier'].unique().tolist()
    eval_metric_score_for_all_dataset = pd.DataFrame(columns=['classifier']+score_dFrame['dataset'].unique().tolist())

    for i, c in enumerate(classifiers): 
        eval_metric_score_for_all_dataset.loc[i] = [c]+score_dFrame[score_dFrame['classifier']==c][metric].tolist()
   

    eval_metric_score_for_all_dataset = eval_metric_score_for_all_dataset.set_index('classifier')
    

    # Rename columns keeping only the number of samples in the dataset
    eval_metric_score_for_all_dataset = eval_metric_score_for_all_dataset.set_axis([int(s.split('_')[0]) for s in eval_metric_score_for_all_dataset.columns.tolist() ], axis=1)
    eval_metric_score_for_all_dataset = eval_metric_score_for_all_dataset.reindex(sorted(eval_metric_score_for_all_dataset.columns), axis=1)

    fig, ax = plt.subplots()
    sns.heatmap(eval_metric_score_for_all_dataset, 
                annot=True,
                linewidth=.5,
                cmap="RdBu", 
                vmax=1,
                vmin=0,
                annot_kws={'size': 9},
                cbar_kws={'label': 'Balanced Accuracy'}).set(xlabel='Number of samples', ylabel='classifier')

    plt.yticks(fontsize=11)
    plt.xlabel('Dataset size', fontsize=11)
    plt.ylabel('Classifier', fontsize=11)
    fig.tight_layout()
    

    if isinstance(save_path_img, str):
        fig.savefig(save_path_img, bbox_inches='tight', dpi=150)
    else:
        plt.show()


def get_outperforming_heatmap(data, metric_score, save_path_img=None): 

    data = data.groupby(['classifier', 'dataset'])[metric_score].mean().reset_index()
    
    model_tourneys = defaultdict(int)
    all_models = sorted(data['classifier'].unique())
     
    all_datasets = sorted((data['dataset'].unique()))

    length_dataset_list=len(all_datasets)
    cls_length=len(all_models)

 
    for dataset, group_dataset in data.groupby('dataset'):
        group_dataset.loc[:, metric_score]= group_dataset[metric_score].values / group_dataset[metric_score].max()
       
        group_dataset = group_dataset.set_index('classifier')[metric_score].to_dict()
       
        for (model1, model2) in combinations(group_dataset.keys(), 2):

            if group_dataset[model1] >= group_dataset[model2]:
                model_tourneys[(model1, model2)] += 1
                
            elif group_dataset[model2] >= group_dataset[model1]:
                model_tourneys[(model2, model1)] += 1

    i=0
    
    success_rate_values=[]

    for model1 in all_models:
        j=0
        j=j+i
        for model2 in all_models:
            if model1 == model2:   
                continue
            i= model_tourneys[(model1, model2)]-1
            i= i+1
            j=i+j
           
        success_rate=round(j/(length_dataset_list*len(all_models)-1)*100,2)
         
        success_rate_values.append(success_rate)
        success_values_row=success_rate_values

    success={'Classifier':all_models,   'Average Success Rate':success_values_row}

    columns_names=['Classifier', 'Average Success Rate']
    df=pd.DataFrame(success, columns=columns_names)
    df.sort_values(by=['Average Success Rate'], inplace=True, ascending=False,)

    print('**********************')
    print()
    print('CLASSIFIERS = ',all_models)
    print()
    print('DATASETS = ', all_datasets)
    print()
    print()

    model_tourney_matrix = []
    for pair in list(product(all_models, all_models)):
        model_tourney_matrix.append(model_tourneys[pair])
 
    model_tourney_matrix = np.array(model_tourney_matrix).reshape(cls_length,cls_length)


    mask_matrix = []   
    for x in range(cls_length):
        for y in range(cls_length):
            mask_matrix.append(x == y)
    
    mask_matrix = np.array(mask_matrix).reshape(cls_length, cls_length)


    fig=plt.figure(figsize=(18, 18)) 

    heatmap = sns.heatmap(np.round(model_tourney_matrix /length_dataset_list, 2), fmt='0.0%',
               mask=mask_matrix,
               cmap=sns.cubehelix_palette(500, light=0.95, dark=0.15),
               square=True, annot=True, cbar=False, annot_kws={'size': 26, 'rotation':45}, vmin=0., vmax=1.0,
               xticklabels=[utils.clfs_names_short_mapping[x] for x in all_models], 
               yticklabels=[utils.clfs_names_short_mapping[x] for x in all_models])
    

    ld=length_dataset_list
    plt.xticks(fontsize=29, rotation=45)
    plt.yticks(fontsize=29, rotation=45)
    plt.xlabel('Losses', fontsize=32)
    plt.ylabel('Wins', fontsize=32)
    """plt.title('The score is '+ metric_score + 
              ' : Percentage of  datasets where model A ("Wins")  outperformed model B (Losses) out of %i datasets \n' %ld
              , fontsize=15)"""
    
    #plt.subplots_adjust(bottom=0.3)
    #plt.tight_layout()

    if isinstance(save_path_img, str):
        fig.savefig(save_path_img, bbox_inches='tight', dpi=150)
    else:
        plt.show()


def get_winning_classifier(data, output_directory, metric_score):    # Average score on the set of all classifiers

    # data are ordered in the following way: 
    # first "classifier" (alphabertic order) and application of the classifier to the n dataset 
    rows=[]
    data = data.groupby(['classifier', 'dataset'],axis=0)[metric_score].mean().reset_index()

    data = data.groupby(['classifier', 'dataset'])[metric_score].mean().reset_index()
    
    
    all_models = np.asarray((data['classifier'].unique()))
   
    #sorted_classifiers
    data[metric_score] = data[metric_score].apply(lambda x: round(x, 3))
   
    # product of the number of the datasets in a folder by the total number of classifiers
    length_dataset_list = len(set(data.dataset))
    initial_set =- length_dataset_list
    final_set = 0
    max_mean_score = 0
    score_total = []
    score_std = []
  
      
    for i in range(1,len(all_models)+1):
       
        initial_set=initial_set+length_dataset_list
        final_set=final_set+length_dataset_list
        
        score_val=round(sum(data[metric_score][initial_set:final_set]/length_dataset_list),3)

        score_total.append(score_val)     
        score_std.append(np.std(data[metric_score][initial_set:final_set])) 

        
        max_mean_score=np.max([max_mean_score, score_val])
        if max_mean_score==score_val:
            name_classifier=all_models[i-1]
            best_classifier=name_classifier
            index=i-1
        else:
            name_classifier=best_classifier
            index=index
           

    print('----------------- AVERAGE ' + metric_score.upper() +' on the class of ALL DATASETS---------------------------')
    print('    ')
    for i in range(1,len(all_models)+1):
        print(all_models[i-1],' = ', score_total[i-1],'±',round(score_std[i-1],3))
        row=[all_models[i-1] + ' = '  +  str(score_total[i-1]) + ' ± ' + '' + str(round(score_std[i-1],3))    ]
        rows.append(row)
            

    winner=[name_classifier,max_mean_score,round(score_std[index],3)]
    print(     '    ')
    print('The BEST CLASSIFIER is', name_classifier, 'with ' + metric_score.upper() + ' = ', max_mean_score,
          '±',round(score_std[index],3))       
    print('----------------------------------------------------------------------------------------------')
    
    winner=['The BEST CLASSIFIER is', name_classifier, 'with ' + metric_score.upper() + ' = ', max_mean_score,
          '±',round(score_std[index],3)]
    
    rows=[['----------------- AVERAGE ' + metric_score.upper() +' on the class of ALL DATASETS-----------------------']] +rows \
          + [[str('--------------------------------------------------------------------------------------------------')]] + \
         [['The BEST CLASSIFIER is ' + name_classifier + ' with ' + metric_score.upper() + ' = ' + '' + str(max_mean_score) +
          ' ± ' + '' + str(round(score_std[index],3))]]

   
    
    with open(output_directory + '/winner_average'+'_'+metric_score+'.txt', "w") as output: 
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(rows) 
       
    # Save code for latex tab
    utils.save_latex_table_from_winner_avg(output_directory + '/winner_average'+'_'+metric_score+'.txt')
    


def get_conf_mat(experiment, save_path_img=None):

    if  experiment in ['2q_chsh', '2q_mer', '3q_mer', '3q_svet']:
        problem_str = 'nonLoc'

    elif experiment in ['2_qubits_2_labels','3_qubits_3_labels', '4_qubits_full_5_labels', '5_qubits_full_7_labels']:
        problem_str = 'efs'
    
    else:
        print('Not valid test', experiment)
        return
    
    cm_path_list = glob.glob(os.path.join(os.getcwd(), experiment, 'confusion_matrices', '*.npy'), recursive=True) 

    cm_path_list = [x for x in cm_path_list if 'PGM' in x]
    cm_path_list = [x for x in cm_path_list if '10000' in x or '20000' in x or '30000' in x]


    fig, axes = plt.subplots(1, 3, figsize=(32, 8), sharey=True)
    titles = ['a','b','c']

    for idx, (cmPath, title) in enumerate(zip(sorted(cm_path_list), titles)):
        cm_save_path = cmPath[:-4]
        cm_with_labels = np.load(cmPath)
        
        cm = cm_with_labels[:, :-1].astype(int)
        labels = cm_with_labels[:, -1]

        font_size_per_nClasses = {2:'36', 3:'30', 5:'24', 7:'18'}
        font_size = int(font_size_per_nClasses[len(labels)])

        str_labels = utils.get_str_labels(labels, problem_str)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        assert np.round(np.sum(cm)) == float(len(labels))  # check if every row of conf matrix sum up to one
    
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                    linewidth=.5, cbar=(idx == 2), ax=ax, vmax=1, vmin=0,
                    annot_kws={'size': font_size-3},
                    cbar_kws={'label': 'Percentage of samples'})
        

        # Adjust the colorbar if present
        if idx == 2:
            cbar = ax.collections[0].colorbar
            cbar.ax.set_ylabel('Percentage of samples', fontsize=font_size)
            cbar.ax.tick_params(labelsize=font_size-2)

        ax.set_xlabel('Predicted label', fontsize=str(font_size))
        if idx==0: ax.set_ylabel('True label', fontsize=str(font_size))
        ax.set_xticklabels(str_labels, fontsize=str(font_size-2))
        ax.set_yticklabels(str_labels, fontsize=str(font_size-2))
        ax.set_title(title, fontweight='bold', fontsize=str(font_size))
        

    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.09)
    
    if isinstance(save_path_img, str):
         plt.savefig(save_path_img+'.png', bbox_inches='tight', dpi=150)
    else:
        plt.show()

   
    



def get_all_stats(tests):
    for experiment in tests:
        print("".join(['#']*5)+experiment)

        parent_dir = os.getcwd()

        path=parent_dir+'/'+experiment

        metric='balanced_accuracy'  
        metric_file=path+'/score_files/'+metric+'.csv'


        score_dFrame = pd.read_csv(metric_file,
                                        sep=',', 
                                        names=['dataset','classifier', metric, 'best_params', 'exc_time'], index_col=False).fillna('')  


        # Function calls
        get_winning_classifier(score_dFrame, path, metric)
        get_outperforming_heatmap(score_dFrame, metric, save_path_img=path+'/'+experiment+'_outperforming_clf')   
        get_classifier_vs_dataset_heatmap(score_dFrame=score_dFrame, metric=metric, save_path_img=path+'/'+experiment+'_clfs_vs_datasets') 
        get_conf_mat(experiment, save_path_img=path+'/'+experiment+'_conf_mat')



if __name__ == "__main__":
    get_all_stats(['2_qubits_2_labels'])
