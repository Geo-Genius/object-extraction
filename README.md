Usage
======
### object_extraction train-mosaic --help
```
Usage: object_extraction train-mosaic [OPTIONS]

  Use AI Model to Extract Object.

Options:
  --img_paths TEXT         image's obs path for train
  --label_paths TEXT       label's obs path for train
  --div_ratio TEXT         train/val dataset div ratio
  --pre_train_model TEXT   pre_train_model's obs path for train
  --epochs TEXT            train epochs
  --output_path, --o TEXT  the output path for result
  --help                   Show this message and exit.
```
* examples:
```
object_extraction train-mosaic --img_paths ${img_paths} --label_paths ${label_paths} --div_ratio ${div_ratio} --pre_train_model ${pre_train_model} --epochs ${epochs} --output_path ${output_path}
```

Usage
======
### object_extraction predict-mosaic --help
```
Usage: object_extraction predict-mosaic [OPTIONS]

  Use AI Model to Extract Object.

Options:
  --model_path TEXT        obs model path
  --cat_ids TEXT           image's catalog ids for predict
  --paths TEXT             image's obs path for predict
  --output_path, --o TEXT  the output path for result
  --help                   Show this message and exit.
```
* examples:
```
object_extraction predict-mosaic --model_path ${model_path} --paths ${paths} --output_path ${output_path}
```






