# How to use

# 1.Prepare Dataset
- unzip wheat.zip to `/your/data/path`
- set `/your/data/path` in configs/wheat.yaml

# 2.Train
```
python src/train.py --config-file configs/wheat.yaml
```

# 3.Test
- set your model path and data path in `src/test.py`
```
python src/test.py
```

# 4.plot 
```
python src/plot.py
```