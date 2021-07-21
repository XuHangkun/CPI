# CPI

## Train the model
```bash
$ #you can find the the train scripts in directory scripts
$ python run_train.py --model_name [baseline,transformercpi] --evt_num EVT_NUM
```

## Test the model
```bash
$ # change the model at inference.py first
$ python run_test.py
``` 

## Draw the train info
```bash
$ # you can use the python file in analysis to analysis the train info 
$ # and evaluate the ROC,PRC,ACU of your trained model
$ python ./analysis/draw_train_info.py --input FILE_PATH
```

