# Person recognition based on the ear

Tool for training ResNeXt101 deep neural network architecture to identify person based on the ear.

## Abstract

The aim of this thesis is to confirm the hypotheses that a neural network trained on the
left ears will be able to recognize subjects by their right ear and vice versa.
Another task was to find a suitable model to prove or disprove the hypotheses.
Finally, experimentation and analysis of the results obtained confirmed the hypothesis on the selected model on 91%.

## Used libraries

- Scikit-learn 1.0.2
- Yaml 5.3.1
- Imgaug 0.4.0 - ( https://github.com/aleju/imgaug )
- Matplotlib 3.5.2
- Numpy 1.22.3
- Tensorflow 2.8.0
- Tensorflow-addons 0.16.1
- Torchvision 0.12.0+cu102
- Torch 1.11.0+cu102
- Keras 2.8.0
- PIL 9.1.0
- Classification_models 1.0.0

## Other libraries

- Os
- Random
- Shutil
- Sys
- Getopt
- Datetime
- Itertools
- Argparse

## Training

Training convolutional neural network with CUDA v11.2 with CUPTI

## Configuration file `conf.yml`

Name of the configuration file must have name `conf.yml`

### Structure of configuration file

- generator
  - batch_size: int
  - image_height: int
  - image_width: int
  - validation_split: float
- training
  - epochs: int
  
### Doc to configuration file

**generator** - is used for setting a generator of Tensorflow library <br>
  **batch_size** - the size of batch to be produced by the generator (default: 50) <br>
  **image_height** - the image height at which the input data should be read (default: 257px) <br>
  **image_width** - the image width at which the input data should be read (default: 161px) <br>
  **validation_split** - distribution of the validation set of training samples  <br>
**training** - sets the number of training epochs. Other parameters are calculated during the training process automatically from the dataset. <br>
**epochs** - number of model training epochs (default: 150) <br>

## Program usability

It is neccessary to download sorted dataset `sorted-dataset` available at [here](https://drive.google.com/file/d/1Q3agDJwyJg-dPjNDIj3ee0PWDjGY8tWX/view?usp=sharing).

Tool uses several switches:

`--split-whole <path/to/dataset>` - split dataset into 60% of training and 40% of testing sets. Both sets contain samples of right and left ears. New dataset folder is created in folder, where `main.py` is located.

`--split-left-right <path/to/dataset>` - split dataset into 60% of right ears (for training) and 40% of another right ears (for testing). At the same time it splits dataset into 60% of left ears (for training) and 40% of rest of left ears (for testing). In result there are 4 folders within dataset.

`training <path/to/dataset>` - runs training process of neural network with settings from configuration file `conf.yml`
Result of training process is then available at `/<path/to/project>/result/<current_time>/`
Result of training process contains:
  - tensorboard file with tracked training process
  - saved trained model
  - loss function curves and accuracy during the training process
  - ROC curve
  
`--to-train [left | right]` - specifies the set we want to train. Only makes sense when using training dataset **splitted-data**. Otherwise it is ignored. In case of using dataset which has subfolders `/train` and `/test`, the following are selected.

`--to-test [left | right]` - similar as **--to-train** it allows to tell the program to take a sample of either right or left ears. Use only has effect when using `splitted-data` dataset. When using the dataset splitted-data this switch must be defined.. otherwise an error occurs.

`--evaluate <saved-model> <path/to/dataset` - it is used for evaluating trained model. It is neccessary to set switch **--to-test** to either on right of left. Depends on which site we want to test. **--to-test** switch needs to define before **--evaluate**. Switch **--to-test** only makes sense if **splitted-data** dataset is used.

## Samples of usage

1. Shows user manual
```bash
python3 main.py -h
```

2. 
```bash
python3 main.py --split-whole /path/to/project/sorted-data/Images
```

3.
```bash
python3 main.py --split-left-right /path/to/project/sorted-data/Images
```

4.
```bash
python3 main.py --training /path/to/dataset/
```

5. Usage of **splitted-data** dataset. Training on left samples, testing on right.
```bash
python3 main.py --training /path/to/project/splitted-data/ --to-train left --to-test right
```

6. Usage of **splitted-data** dataset. Switch **--to-test** on left ears. Start testing on trained model Produces ROC curve in /path/to/project/results/<current_time>/ 
```bash
python3 main.py --to-test left --evaluate /path/to/saved-model/ /path/to/project/splitted-data
```
                 








