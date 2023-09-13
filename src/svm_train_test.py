import os
import random
import numpy as np
from sklearn.svm import SVC

import time
from multiprocessing import Pool

random.seed(12)

def worker(args):
    
    t_start = time.time()
    print("开始执行,进程号为%d" % (os.getpid()))

    task, fn = args
    X_trn = np.load(os.path.join(root, fn))
    label_name = fn.replace('.npy','_label.npy').replace(f'_{task}','')
    Y_trn = np.load(os.path.join(root, label_name))

    clf = SVC(probability=True, kernel='linear')
    clf.fit(X_trn, Y_trn)

    X_tst = np.load(os.path.join(root, fn.replace('_bal','').replace('train','test')))
    Y_tst = np.load(os.path.join(root, label_name.replace('_bal','').replace('train','test')))

    res = clf.predict_proba(X_tst)

    pred_y = np.argmax(res, axis=1)
    mask = pred_y==Y_tst
    print(f'{fn} acc:', sum(mask)/len(mask))

    with open(f'./results/svm_{fn[:-4]}.txt', 'w+') as f:
        for ix in range(len(res)):
            line = fn + ' ' + str(Y_tst[ix]) + ' ' + ' '.join(list(map(str, res[ix]))) + '\n'
            f.write(line)
    
    print(f'{task} {fn} was done!')
    t_stop = time.time()
    print(os.getpid(), "执行完毕，耗时%0.2f" % (t_stop-t_start))


if __name__ == "__main__":

    root = '/opt/data1/dyw/GrainSet/features'

    xx = [el.replace('svm_','').replace('.txt','.npy') for el in os.listdir('/home/dyw/code/Grainset/results/svm')]

    all_fns = []
    for task in ['hist', 'sift']:
        for fn in os.listdir(root):
            if 'train' in fn and task in fn and fn not in xx:
                all_fns.append((task,fn))

    nw = len(all_fns)

    print('Total nw:', nw)

    po = Pool(min(nw,8)) 
    for i in range(nw):
        po.apply_async(worker, (all_fns[i],))

    print("----start----")
    po.close()
    po.join()
    print("-----end-----")

