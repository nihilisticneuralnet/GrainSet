# Overview

This is repository for our *Nature Scientific Data* paper: **An annotated grain kernel image database for visual quality inspection**. (DOI: https://doi.org/10.1038/s41597-023-02660-8)


# Relased Datasets

We released four types of cereal grains: Wheat, Maize, Sorghum and Rice in single-kernel images with experts' annotations. Additionally, 

- **GrainSet-tiny**: this is a preview for understanding our database by randomly selecting 2% samples from GrainSet.
- **GrainSet-raw**: this is a reference for understanding the data acquisition and pre-processing procedures by randomly selecting 5% raw images captured by our acquisition device.

|Species|Num.|URL|
|---|----|----|
|Wheat|200K| https://doi.org/10.6084/m9.figshare.22992317.v2|
|Maize|19K|	https://doi.org/10.6084/m9.figshare.22987562.v2|
|Sorghum| 102K| https://doi.org/10.6084/m9.figshare.22988981.v2|
|Rice|31K|https://doi.org/10.6084/m9.figshare.22987292.v3|
|GrainSet-tiny|6.5K|https://doi.org/10.6084/m9.figshare.22989029.v1|
|GrainSet-raw|15K|https://doi.org/10.6084/m9.figshare.24137472.v1|




# Validation

### 1.Prepare Datasets

- `unzip` wheat/maize/sorghum/rice.zip to `/your/data/path`
- download **datalist.zip** from datasets folder
-  `unzip` **datalist.zip** to runs/datalist

### 2.Train deep learning-based Models

- set `data_path`` and `CUDA_VISIBLE_DEVICES`` in **.sh** files
- run shell scripts, *e.g.*: ```bash run_res50.sh```

### 3.Train traditional SVM


- extract features: ``` python src/extract_feature.py```
- train svm classifier: ```python src/svm_train_test.py```
- library supports:
      
```
      python==3.7     
      opencv-contrib-python==3.4.2.17     
      opencv-python==3.4.2.17
```

### 4.Test

- set your model path and data path in `src/test.py`
- run test: ```python src/test.py ```

### others

- plot figures: ```python src/plot.py```



# Citation

If our paper has been of assistance, we would appreciate it if you could consider citing it in your work.

```
@article{fan2023annotated,
  title={An annotated grain kernel image database for visual quality inspection},
  author={Fan, Lei and Ding, Yiwen and Fan, Dongdong and Wu, Yong and Chu, Hongxia and Pagnucco, Maurice and Song, Yang},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={778},
  year={2023},
  publisher={Nature Publishing Group UK London}
}


@inproceedings{fan2022grainspace,
  title={GrainSpace: A Large-scale Dataset for Fine-grained and Domain-adaptive Recognition of Cereal Grains},
  author={Fan, Lei and Ding, Yiwen and Fan, Dongdong and Di, Donglin and Pagnucco, Maurice and Song, Yang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21116--21125},
  year={2022}
}


@incollection{fan2023ai4graininsp,
  title={Identifying the Defective: Detecting Damaged Grains for Cereal Appearance Inspection},
  author={Fan, Lei and Ding, Yiwen and Fan, Dongdong and Wu, Yong and Pagnucco, Maurice and Song, Yang},
  booktitle={ECAI 2023},
  year={2023},
  publisher={IOS Press}
}


@article{fan2023av4gainsp,
  title={AV4GAInsp: An Efficient Dual-Camera System for Identifying Defective Kernels of Cereal Grains},
  author={Fan, Lei and Ding, Yiwen and Fan, Dongdong and Wu, Yong and Chu, Hongxia and Pagnucco, Maurice and Song, Yang},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}

```