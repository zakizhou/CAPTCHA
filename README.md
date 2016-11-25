# CAPTCHA
This repo is a tutorial for training a cnn model for recognizing
numbers of captcha

##USAGE
Before run training, training images and validation images should be 
 generated, change to the root dir of this project and run
 ```
python CNN/captcha_producer.py -n 50000 -p "images/train"
 ```
this will generate 50000 training images in the images/train/

for validation set:
 ```
python CNN/captcha_producer.py -n 10000 -p "images/validation"
 ```
make sure you have make those dir already

For running:
 ```
python CNN/captcha_train.py
 ```

I will soon add multi-gpus training to this repo

##Benchmark
After 40 minutes training on single tesla K20m, this model achieved 
around 86% accuracy, adjusting the scale of parameters or adding dropout
should still improve this performance
