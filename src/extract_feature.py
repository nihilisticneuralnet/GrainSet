import os
import numpy as np
import cv2
import time
from multiprocessing import Pool

def worker(fn):

    t_start = time.time()
    print("开始执行,进程号为%d" % (os.getpid()))

    all_hist = []
    all_sift = []
    all_label = []

    with open(os.path.join(txt_root, fn),'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n','')
            path, label = line.split(' ')
            img = cv2.imread(os.path.join(data_root, fn.split('_')[0], path))
            img = cv2.resize(img, (224,224))

            hists = []
            try:
                for i in range(3):
                    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
                    hists.append(histr)
                hists = np.concatenate(hists).reshape(-1)

                sift = cv2.xfeatures2d.SIFT_create()
                kp, des = sift.detectAndCompute(img, None)   #des是描述子
                des = des.reshape(-1)[:sift_dim]
                if len(des)<sift_dim:
                    des = np.pad(des, (0,sift_dim-len(des)))
            except:
                print(f'ERROR {fn} {path}')
                hists = np.random.randint(0,100,768).astype(np.float32)
                des   = np.random.randint(0,100,sift_dim).astype(np.float32)

            all_hist.append(hists)
            all_sift.append(des)
            all_label.append(int(label))

    all_hist  = np.stack(all_hist, axis=0)
    all_sift  = np.stack(all_sift, axis=0)
    all_label = np.array(all_label)

    np.save('/opt/data1/dyw/GrainSet/features/' + fn.replace('.txt','_hist.npy'), all_hist)
    np.save('/opt/data1/dyw/GrainSet/features/' + fn.replace('.txt','_sift.npy'), all_sift)
    np.save('/opt/data1/dyw/GrainSet/features/' + fn.replace('.txt','_label.npy'), all_label)

    print(f'{fn} was done!')

    t_stop = time.time()
    print(os.getpid(), "执行完毕，耗时%0.2f" % (t_stop-t_start))


if __name__ == "__main__":

    sift_dim = 50*128
    data_root = '/opt/data1/dyw/GrainSet'
    txt_root = 'runs/datalist'
    all_fns = os.listdir(txt_root)
    nw = len(all_fns)

    po = Pool(nw) 
    for i in range(nw):
        po.apply_async(worker, (all_fns[i],))

    print("----start----")
    po.close()
    po.join()
    print("-----end-----")


