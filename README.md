Docker build step:
=====
1. fetch target branch code:
```
git clone ssh://git@code-cbu.huawei.com:2233/geogenius/geogenius-tools.git
cd geogenius-tools
git fetch origin master:master
git checkout master
```
2. package the object-extraction tools:
```
cd geogenius-tools
tar -zcvf object-extraction.tar.gz object-extraction/
```

3. put `object-extraction.tar.gz` and `Dockerfile` under same empty folder, for example:
```
mkdir -p /opt/docker
cp object-extraction.tar.gz /opt/docker/
cp object-extraction/Dockerfile /opt/docker/
```
4. build object-extraction image:
```
cd /opt/docker/
```
build command:
```
FROM swr.cn-north-5.myhuaweicloud.com/geogenius_dev/geogenius_sdk:v1.0
ADD ./object-extraction.tar.gz /opt
ENV PATH ${PATH}:/root/anaconda3/bin/
WORKDIR /opt/object-extraction
RUN ["/root/bin/python", "setup.py", "install"]
```
5. upload the image to swr:
login swr:
```
docker login -u cn-north-5@xxx -p xxx swr.cn-north-5.myhuaweicloud.com
```
tag image:
```
docker tag object-extraction:v0.1 swr.cn-north-5.myhuaweicloud.com/geogenius_dev/object-extraction:v0.1 
```
push image to swr:
```
docker push swr.cn-north-5.myhuaweicloud.com/geogenius_dev/object-extraction:v0.1
```


Docker env
======
follow environments need to be set when execute Dockerfile:
```
ACCESS_KEY=ak
SECRET_KEY=secret key
```

Usage
======
### rio object-extraction predict-mosaic --help
```
UUsage: rio object-extraction train-mosaic [OPTIONS]

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
rio object-extraction predict-mosaic --model_path ${model_path} --paths ${paths} --output_path ${output_path}
```

## Json Doc
注册工具body
```json
{
	"name": "模型推理-mosaic",
	"iconUrl": "https://geogenius-bucket.obs.cn-north-5.myhuaweicloud.com/geogenius-config/geogenius-image/geogenius/icon2.jpg",
	"description": "desc",
	"toolType": "Docker",
	"labels": ["MosaicImage"],
	"executorEngineDescriptor": {
		"cmd": "rio object-extraction predict-mosaic --model_path ${model_path} --paths ${paths} --output_path ${output_path}",
		"docker": {
			"image": "swr.cn-north-5.myhuaweicloud.com/geogenius_dev/object-extraction:v41"
		}
	},
	"inputParameters": [{
		"name": "model_path",
		"paramType": "STRING",
		"description": "模型存储位置",
		"isRequired": true,
		"multiple": false
	}, {
		"name": "paths",
		"paramType": "STRING",
		"description": "obs 路径列表",
		"isRequired": true,
		"multiple": false
	}],
	"outputParameters": [{
		"name": "output_path",
		"paramType": "STRING",
		"description": "图像输出位置",
		"isRequired": true,
		"multiple": false
	}]
}
```


Usage
======
### rio object-extraction train-mosaic --help
```
Usage: rio object-extraction predict-mosaic [OPTIONS]

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
rio object-extraction predict-mosaic --model_path ${model_path} --paths ${paths} --output_path ${output_path}
```

## Json Doc
注册工具body
```json
{
	"name": "地物分类-训练-mosaic",
	"iconUrl": "https://geogenius-bucket.obs.cn-north-5.myhuaweicloud.com/geogenius-config/geogenius-image/geogenius/icon2.jpg",
	"description": "desc",
	"toolType": "Docker",
	"labels": ["MosaicImage"],
	"executorEngineDescriptor": {
		"cmd": "rio object-extraction train-mosaic  --img_paths ${img_paths} --label_paths ${label_paths} --div_ratio ${div_ratio} --output_path ${output_path} --epochs ${epochs} --pre_train_model ${pre_train_model}",
		"docker": {
			"image": "swr.cn-north-5.myhuaweicloud.com/geogenius_dev/object-extraction:v47"
		}
	},
	"inputParameters": [{
		"name": "epochs",
		"paramType": "STRING",
		"description": "训练epochs数量",
		"isRequired": true,
		"multiple": false,
		"defaultValue": ["50"]
	}, {
		"name": "pre_train_model",
		"paramType": "STRING",
		"description": "预训练模型存放位置",
		"isRequired": true,
		"multiple": false
	}, {
		"name": "img_paths",
		"paramType": "STRING",
		"description": "训练/验证集存储位置",
		"isRequired": true,
		"multiple": false
	}, {
		"name": "label_paths",
		"paramType": "STRING",
		"description": "训练/验证标签存储位置",
		"isRequired": true,
		"multiple": false
	}, {
		"name": "div_ratio",
		"paramType": "STRING",
		"description": "训练集/验证集划分比例",
		"isRequired": true,
		"multiple": false,
		"defaultValue": ["0.8"]
	}],
	"outputParameters": [{
		"name": "output_path",
		"paramType": "FILE",
		"description": "模型输出路径",
		"isRequired": false,
		"multiple": false,
		"isMount": true,
		"mountPath": "/var/lib/docker/output"
	}]
}
```

