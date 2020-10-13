# Linear Separability Evaluation
This repo provides the scripts to test a learned AlexNet's feature representation performance at the five different convolutional levels -- in parallel. The training lasts 36 epochs and should be finished in <1.5days.



## Usage
`$python eval_linear_probes.py`
```
usage: eval_linear_probes.py [-h] [--data DATA] [--ckpt-dir DIR] [--device d]
                             [--modelpath MODELPATH] [--workers N]
                             [--epochs N] [--batch-size N]
                             [--learning-rate FLOAT] [--tencrops] [--evaluate]
                             [--img-size IMG_SIZE] [--crop-size CROP_SIZE]
                             [--imagenet-path IMAGENET_PATH]

AlexNet standard linear separability tests

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Dataset Imagenet or Places (default: Imagenet)
  --ckpt-dir DIR        path to checkpoints (default: ./test)
  --device d            GPU device
  --modelpath MODELPATH
                        path to model
  --workers N           number of data loading workers (default: 6)
  --epochs N            number of epochs (default: 36)
  --batch-size N        batch size (default: 192)
  --learning-rate FLOAT
                        initial learning rate (default: 0.01)
  --tencrops            flag to not use tencrops (default: on)
  --evaluate            flag to evaluate only (default: off)
  --img-size IMG_SIZE   imagesize (default: 256)
  --crop-size CROP_SIZE
                        cropsize for CNN (default: 224)
  --imagenet-path IMAGENET_PATH
                        path to imagenet folder, where train and val are
                        located
```

## Settings
The settings follow the caffe code provided in [Zhang et al.](https://github.com/richzhang/colorization), with optional tencrops enabled. Average pooling can be used, but max-pooling is faster and overall more common so it is used here. 


## Reference

If you use this code, please consider citing the following paper:

Yuki M. Asano, Christian Rupprecht and Andrea Vedaldi.  "A critical analysis of self-supervision, or what we can learn from a single image." Proc. ICLR (2020)

```
@inproceedings{asano2020a,
  title={A critical analysis of self-supervision, or what we can learn from a single image},
  author={Asano, Yuki M. and Rupprecht, Christian and Vedaldi, Andrea},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020},
}
```
