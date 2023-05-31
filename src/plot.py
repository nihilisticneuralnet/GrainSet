import numpy as np
import os
from sklearn.metrics import confusion_matrix
from utils.util import plot

from openTSNE import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


name_dic = {
        'wheat':['NOR','F&S','SD','MY','AP','BN','BP', 'IM', 'Recall','Accuracy','F1-score'], 
        'maize':['NOR','F&S','SD','MY','AP','BN','HD', 'IM', 'Recall','Accuracy','F1-score'],
        'sorg':['NOR','F&S','SD','MY','AP','BN','HD',  'IM', 'Recall','Accuracy','F1-score'],
        'rice':['NOR','F&S','SD','MY','AP','BN','UN', 'IM','Recall','Accuracy','F1-score']}

def plot_res(grain_t='wheat'):
    
    all_label = []
    embeds = []
    files = []
    all_pred = []

    for line in open(f'./results/{grain_t}_prob.txt','r').readlines():
        line = line.replace('\n','')
        content = line.split(' ')
        files.append(content[0])
        all_label.append(int(content[1]))
        embeds.append(list(map(float, content[2:])))
    
    embeds = np.vstack(embeds)
    all_label = np.hstack(all_label)
    all_pred = np.argmax(embeds, axis=1) 

    files = np.hstack(files).tolist()
    tsne_xy = TSNE().fit(embeds)

    label_name_dic = {ix:el for ix, el in enumerate(name_dic[grain_t][:8])}
    fig,ax = plt.subplots(figsize=(8, 5))
    tsne_label = [label_name_dic[el] for el in all_label] 

    ax.set_xlim(xmin=-90,xmax=90)
    ax.set_ylim(ymin=-80,ymax=80)

    plot(tsne_xy, tsne_label, ax, colors=None, label_order=name_dic[grain_t][:8])

    fig.savefig(f'./results/tsne_{grain_t}.jpg', dpi=300)


    sns.set()
    fig,ax = plt.subplots()
    yt = [name_dic[grain_t][ix] for ix in all_label]
    yp = [name_dic[grain_t][ix] for ix in all_pred]

    C2 = confusion_matrix(yt, yp, labels=name_dic[grain_t])
    C2 = C2.astype(np.float)
    recall = [C2[ix][ix]/sum(C2[ix,:]) for ix in range(8)]
    precision = [C2[ix][ix]/sum(C2[:,ix]) for ix in range(8)]
    f1 = [2*recall[ix]*precision[ix]/(precision[ix]+recall[ix]+1e-6) for ix in range(8)]

    f1 = [round(f1[ix]*1000)/10 for ix in range(8)]
    recall = [round(recall[ix]*1000)/10 for ix in range(8)]
    precision = [round(precision[ix]*1000)/10 for ix in range(8)]
    # acc = [round(acc[ix]*1000)/10 for ix in range(8)]

    C2[:,8][:8] = recall
    C2[:,9][:8] = precision
    C2[:,10][:8] = f1
    C2 = C2[:8]
    df=pd.DataFrame(C2,index=name_dic[grain_t][:8],columns=name_dic[grain_t])
    sns.set(font_scale=1.2)
    plt.rc('font',size=10)
    sns.heatmap(df, annot=True, cbar=True, fmt='.5g', cmap="GnBu",  vmin=0, vmax=400, center=200, square=True, linewidths=1, linecolor='black', cbar_kws={"shrink": 0.8}) #cmap="Reds", # annot_kws={'color':'black'},
    

    ax.tick_params(right=False, top=False, labelright=False, labelbottom=False, labelrotation=90, labeltop=True, labelleft=True) # labelrotation=45

    plt.yticks(rotation=0)


    # ax.set_title('confusion matrix') #标题
    # ax.set_xlabel(f'R50 on {grain_t} data') #x 轴
    # ax.set_ylabel('true') #y 轴
    fig.savefig(f'./results/{grain_t}_matrix.png',dpi=300)


for grain_t in ['wheat', 'sorg','maize','rice']:
    plot_res(grain_t)
