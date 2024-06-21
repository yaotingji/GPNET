# GPNET

# Setup and dependencies

Dependencies：

- python 3.10.4
- pytorch 1.12.1
- numpy 1.22.3
- pandas 1.4.3

# Usage

### 1. How to preprocess the raw data
First, You need prepare txt files containing compound labels and point-cloud information.

For labels file, whose format is like below:
```
compound1 label
compound2 label
...
```

For point-cloud files, whose format is like below:
```
x y z esp
...
```

Using npz files to compress and save the above information:
```python
python data_npz.py 
```

## 2. How to train the models. 
Instance the model object and do training.
```python
python GPNET-trainer.py 
```
Loss, AUC-ROC on training set and validation set will be saved as txt files and network parameters after each training epoch will be saved.

## 3. How to do prediction
 ```
    python GPNET.py -o result.csv -t False
   ```
**-o** is the specified results saved file.
**-t** is the specified output format, which defaults to True, exporting the predicted labels. 
If you want to know the predicted binding possibility(score) of compound, you can specify it to False.

