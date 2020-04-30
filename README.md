# Input channel dropout code

## Installation

### With nvidia-docker (recommended)

```bash
docker build -t input_channels_dropout .
``` 

Run mounting downloaded directories (see tfrecords section):

```bash
docker run --rm -it \
    -v /local/input_channel_dropout/multispectral:/input_channel_dropout/multispectral \
    -v /local/input_channel_dropout/faster_rcnn_resnet50_coco_2018_01_28:/input_channel_dropout/faster_rcnn_resnet50_coco_2018_01_28 \
    input_channels_dropout
``` 

### Manual

Please use python>=3.6 and tensorflow 1.xx

[How to install ?](research/object_detection/g3doc/installation.md) (research/object_detection/g3doc/installation.md)


## Get TFRecord datasets

Download directory at `https://1drv.ms/u/s!AkO079ItTrSpaByU2MyIs0WeNPM?e=guajrc`

It contains:

- `faster_rcnn_resnet50_coco_2018_01_28` directory holding pre-trained checkpoint
- `multispectral` directory holding train and validation TFRecords, label_map.pbtxt and config file for training on multispectral dataset.
- `rgbd_pedestrian` directory holding train and validation TFRecords, label_map.pbtxt and config file for training on rgbd dataset.

## Train

Choose DROP_MODE amongst:

- no_drop
- simultaneous_095, simultaneous_09, simultaneous_08, simultaneous_07, simultaneous_06, simultaneous_05
- independent_095, independent_09, independent_08, independent_07, independent_06, independent_05
- rgb_only
- additional_only

If you are using `multispectral` dataset, set N_ADDITIONAL=3, if you are using `rgbd_pedestrian` dataset, set N_ADDITIONAL=1 

Multispectral example:

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`/research
export PYTHONPATH=$PYTHONPATH:`pwd`/research/slim
export N_ADDITIONAL=3
export EVAL_WITH_DROPOUT=False
export CUDA_VISIBLE_DEVICES=1
export DROP_MODE=independent_05
MODEL_DIR=multispectral/train/independent_05_3
nohup python research/object_detection/model_main.py \
    --pipeline_config_path=multispectral/faster_rcnn_resnet50_multispectral.config \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=200000 \
    --sample_1_of_n_eval_examples=1 \
    --checkpoint_was_trained_on_rgb_only=True \
    --alsologtostderr	&
```

RGBD example:

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`/research
export PYTHONPATH=$PYTHONPATH:`pwd`/research/slim
export N_ADDITIONAL=1
export EVAL_WITH_DROPOUT=False
export CUDA_VISIBLE_DEVICES=2
export DROP_MODE=independent_09
MODEL_DIR=rgbd_pedestrian/train/independent_09_3
nohup python research/object_detection/model_main.py \
    --pipeline_config_path=rgbd_pedestrian/faster_rcnn_resnet50_rgbd_pedestrian.config \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=200000 \
    --sample_1_of_n_eval_examples=1 \
    --checkpoint_was_trained_on_rgb_only=True \
    --alsologtostderr	&
```

## Evaluate

In order to measure underfitting, it is necessary to evaluate the model with RGB channels only or with additional channels only.
For this, set EVAL_WITH_DROPOUT=True and DROP_MODE=rgb_only or DROP_MODE=additional_only

You can also create `predictions.json` and `ground_truth.json` files that can be used for computing channels complementarity.
For this, set `JSON_OUT=${MODEL_DIR}/predictions.json`

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`/research
export PYTHONPATH=$PYTHONPATH:`pwd`/research/slim
export CUDA_VISIBLE_DEVICES=1
export DROP_MODE=rgb_only
export N_ADDITIONAL=3
export EVAL_WITH_DROPOUT=True
MODEL_DIR=multispectral/train/rgb_only_1
export JSON_OUT=${MODEL_DIR}/predictions.json
python research/object_detection/model_main.py \
    --pipeline_config_path=multispectral/faster_rcnn_resnet50_multispectral.config \
    --checkpoint_dir=${MODEL_DIR} \
    --run_once=true \
    --sample_1_of_n_eval_examples=1 \
    --checkpoint_was_trained_on_rgb_only=true \
    --alsologtostderr
```

## Compute complementarity

Once you have trained models with additional channels only and RGB channels only using the `DROP_MODE` option 
and you have evaluated them the same way, with the `JSON_OUT` argument so that `predictions.json` and `ground_truth.json`
are now present in each model directory,
you can run this script: 

```bash
python complementarity.py --models_dir=multispectral/train
```

Every subdirectory in `models_dir` with name starting with 'additional' or 'rgb' will be taken into account.
Each should hold predictions.json and ground_truth.json

