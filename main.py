#!/usr/bin/python

import glob
import os
import random
import shutil
import sys, getopt
import yaml
import cv2 as cv
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import torchvision.transforms as T
import keras.layers as keras_layers
from PIL import Image
from PIL import ImageOps
from classification_models.tfkeras import Classifiers
from keras import losses
from keras.models import Sequential
from datetime import datetime
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
import argparse

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = "/results/" + current_time + "/checkpoints/checkpoint.ckpt"

"""
Normalize images with mean and std of imagenet database 

Source: https://www.tutorialspoint.com/pytorch-how-to-normalize-an-image-with-mean-and-standard-deviation
"""


def normalize_images(image):
    image = image.astype(np.uint8)
    img_tensor = T.ToTensor()(image)
    transform = T.Normalize((0.485, 0.456, 0.406), (0.299, 0.224, 0.225))
    normalized_img_tensor = transform(img_tensor)
    normalized_img = T.ToPILImage()(normalized_img_tensor)
    numpy_image = np.asarray(normalized_img)
    image = numpy_image.astype(np.float32)
    return image


"""
Augment images:
    Rotate in angle <-15, 15>
    Crop 0 - 20% from each size independently, resize to target size
    Gaussian Blur with 50% probability on sigma <0.0, 2.0>
    Gaussian Noise with scale <0, 0.08 * 255>
    Brightness multiply (0.5, 1.5), add (-30, 30)
    Gamma Contrast with gamma <0.5, 1.5>
    Hue and Saturation with value <-45, 45>
    Horizontal flip with 50% probability
"""


def augment_images(image):
    image = image.astype(np.uint8)
    # show_image(image)

    second_augment = iaa.Sequential([
        iaa.Rotate((-15, 15)),
        iaa.Crop(keep_size=True, sample_independently=True, percent=(0.0, 0.2)),
        iaa.Sometimes(
            0.2,
            iaa.GaussianBlur(sigma=(0.0, 2.0))
        ),
        iaa.AdditiveGaussianNoise(scale=(0, 0.08 * 255)),
        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
        iaa.GammaContrast(gamma=(0.5, 1.5)),
        iaa.AddToHueAndSaturation(value=(-45, 45)),
        iaa.Fliplr(0.5),
    ])

    image = second_augment(image=image)
    image = image.astype(np.float32)
    normalized = normalize_images(image)
    numpy_image = np.asarray(normalized)
    image = numpy_image.astype(np.float32)

    # first_augment = iaa.Rotate((-15, 15))
    # image = first_augment(image=image)
    # show_image(image)
    #
    # second_augment = iaa.Crop(keep_size=True, sample_independently=True, percent=(0.0, 0.2))
    # image = second_augment(image=image)
    # show_image(image)
    #
    # third_augment = iaa.GaussianBlur(sigma=(0.0, 4.0), seed=35)
    # image = third_augment(image=image)
    # show_image(image)
    #
    # forth_augment = iaa.AdditiveGaussianNoise(scale=(0, 0.30 * 255), seed=35, loc=0.2)
    # image = forth_augment(image=image)
    # show_image(image)
    #
    # fifth_augment = iaa.MultiplyAndAddToBrightness(mul=(1.0, 2.5), add=(-45, 45), seed=35, random_order=True)
    # image = fifth_augment(image=image)
    # show_image(image)
    #
    # sixth_augment = iaa.GammaContrast(gamma=(0.5, 2.0), per_channel=True, seed=35)
    # image = sixth_augment(image=image)
    # show_image(image)
    #
    # seventh_augment = iaa.AddToHueAndSaturation(value=None)
    # image = seventh_augment(image=image)
    # show_image(image)

    return image


"""
Resize images to appropriate target size (161x257) (width/height) with respect to aspect ratio
Add padding to images if not suitable to target size. Add average colour from imagenet database
"""


def preprocess_images(path_to_image):
    img = Image.open(path_to_image)
    image = ImageOps.contain(img, (161, 257))

    # source: https://www.geeksforgeeks.org/add-padding-to-the-image-with-python-pillow/
    width, height = image.size

    target_width = 161
    target_height = 257

    missing_width = target_width - width
    missing_height = target_height - height

    # if odd number then ceil and second is not
    # 145 / 2 = 72.5 -> 72 and 73 => 73+72=145 .. pixels are always a whole number
    right = missing_width // 2
    left = int(np.ceil(missing_width / 2))
    top = int(np.ceil(missing_height / 2))
    bottom = missing_height // 2

    new_width = width + right + left
    new_height = height + top + bottom

    # average color from image net mean = [0.485, 0.456, 0.406]
    # calculated -> red = 255*0.485; green = 255*0.456; blue = 255*0.406 ==> rounded up (124, 116, 104)
    result = Image.new(image.mode, (new_width, new_height), (124, 116, 104))
    result.paste(image, (left, top))

    # override actual image
    result.save(path_to_image)


"""
Run only once if dataset EarVN1.0 is sorted on left and right ear
Split sorted dataset to train and test folders (for training on left and right ears)

Create 2 folders within /whole-net folder : -> /whole-net/train
                                            -> /whole-net/test

walk_through_images_and_resize() method is applicable after calling this method

Sorted dataset hierarchy:

->Images
  - 001.ALI_HD
    - L
        - [images]
    - R
        - [images]
  - 002.LeDuong_BL
  ...

"""


def split_whole_net_training(dataset_path):
    project_path = os.getcwd()

    train = project_path + "/whole-net/train"
    test = project_path + "/whole-net/test"
    new_path = project_path + "/whole-net"

    if not os.path.exists(new_path):
        os.mkdir(new_path)
        os.mkdir(train)
        os.mkdir(test)

    directory = dataset_path

    for filename in os.listdir(directory):
        sub_directory = directory + '/' + filename
        if os.path.isdir(sub_directory):
            new_train_dir = train + '/' + filename
            new_test_dir = test + '/' + filename
            os.mkdir(new_train_dir)
            os.mkdir(new_test_dir)

            train_images = []
            test_images = []

            for sub_dir in os.listdir(sub_directory):
                if sub_dir == 'R':
                    sub_sub_directory = sub_directory + '/' + sub_dir
                    for image in os.listdir(sub_sub_directory):
                        train_images.append(image)
                    for to_copy in random.sample(train_images, int(len(train_images) * .6)):
                        shutil.copy2(sub_sub_directory + '/' + to_copy, new_train_dir)
                        train_images.remove(to_copy)
                    for to_copy_test in train_images:
                        shutil.copy2(sub_sub_directory + '/' + to_copy_test, new_test_dir)
                    train_images = []
                else:
                    sub_sub_directory = sub_directory + '/' + sub_dir
                    for image in os.listdir(sub_sub_directory):
                        test_images.append(image)
                    for to_copy in random.sample(test_images, int(len(test_images) * .6)):
                        shutil.copy2(sub_sub_directory + '/' + to_copy, new_train_dir)
                        test_images.remove(to_copy)
                    for to_copy_test in test_images:
                        shutil.copy2(sub_sub_directory + '/' + to_copy_test, new_test_dir)
                    test_images = []


"""
Info: Run only once if dataset EarVN1.0 is sorted on left and right ear
Purpose: 
Split dataset to 4 folders so we can train on one side and test with another side
4 folders will be created inside /splitted-data folder -> /splitted-data/left_train
                                                       -> /splitted-data/left_test
                                                       -> /splitted-data/right_train
                                                       -> /splitted-data/right_test

walk_through_images_and_resize() method is applicable after calling this method

Sorted dataset hierarchy:
->Images
  - 001.ALI_HD
    - L
        - [images]
    - R
        - [images]
  - 002.LeDuong_BL
  ...
"""


def split_train_test_dataset(dataset):
    project_path = os.getcwd()

    left_train_path = project_path + "/splitted-data/left_train"
    right_train_path = project_path + "/splitted-data/right_train"
    left_test_path = project_path + "/splitted-data/left_test"
    right_test_path = project_path + "/splitted-data/right_test"
    new_path = project_path + "/splitted-data"

    if not os.path.exists(new_path):
        os.mkdir(new_path)
        os.mkdir(left_train_path)
        os.mkdir(right_train_path)
        os.mkdir(left_test_path)
        os.mkdir(right_test_path)

    directory = dataset

    for filename in os.listdir(directory):  # [001, 002, 003 ... ]
        sub_directory = directory + '/' + filename  # [sorted_dataset_grayscale/Images/001
        if os.path.isdir(sub_directory):  # is directory = True
            new_dir_left_train = left_train_path + '/' + filename
            new_dir_right_train = right_train_path + '/' + filename
            new_dir_left_test = left_test_path + '/' + filename
            new_dir_right_test = right_test_path + '/' + filename
            os.mkdir(new_dir_left_train)
            os.mkdir(new_dir_right_train)
            os.mkdir(new_dir_left_test)
            os.mkdir(new_dir_right_test)

            human_images_left = []
            human_images_right = []

            for sub_dir in os.listdir(sub_directory):  # R / L
                if sub_dir == 'R':
                    sub_sub_directory = sub_directory + '/' + sub_dir  # sorted_dataset_grayscale/Images/001/<type_of_data>
                    for image in os.listdir(sub_sub_directory):
                        human_images_right.append(image)
                    for to_copy in random.sample(human_images_right, int(len(human_images_right) * .6)):
                        shutil.copy2(sub_sub_directory + '/' + to_copy, new_dir_right_train)
                        human_images_right.remove(to_copy)
                    for to_copy_test in human_images_right:
                        shutil.copy2(sub_sub_directory + '/' + to_copy_test, new_dir_right_test)
                    human_images_right = []

                else:
                    sub_sub_directory = sub_directory + '/' + sub_dir  # sorted_dataset_grayscale/Images/001/<type_of_data>
                    for image in os.listdir(sub_sub_directory):
                        human_images_left.append(image)
                    for to_copy in random.sample(human_images_left, int(len(human_images_left) * .6)):
                        shutil.copy2(sub_sub_directory + '/' + to_copy, new_dir_left_train)
                        human_images_left.remove(to_copy)
                    for to_copy_test in human_images_left:
                        shutil.copy2(sub_sub_directory + '/' + to_copy_test, new_dir_left_test)
                    human_images_left = []


"""
Walkthough a splitted dataset and use preprocess_images() function on each image
For directory mapped as /whole-net/<train/test>/[folders of humans]/[images] use only 'whole-net' as input
"""


def walk_through_images_and_resize(dataset):

    for sub_directory in os.listdir(dataset):
        name_of_sub_directory = dataset + '/' + sub_directory
        if os.path.isdir(name_of_sub_directory):
            for person_folder in os.listdir(name_of_sub_directory):
                name_person_folder = name_of_sub_directory + '/' + person_folder
                if os.path.isdir(name_person_folder):
                    for img in os.listdir(name_person_folder):
                        img_path = name_person_folder + '/' + img
                        preprocess_images(img_path)


"""
Scheduled learning rate used for decaying learning rate during training process as callback.
Decaying every epoch
"""


def scheduled_learning_rate(epoch, lr):
    return 0.001 * 1 / (1 + 0.66 * epoch)


"""
Extends convolutional base by new head. Using optimizer and then compile whole model.
@:returns new extended model 
"""


def extend_model_by_head(pretrained_model, number_of_classes=164):
    extended_model = Sequential()
    extended_model.add(pretrained_model)
    extended_model.add(keras_layers.GlobalAveragePooling2D())
    extended_model.add(keras_layers.Dense(2048, activation='relu'))
    extended_model.add(keras_layers.Dense(2048, activation='relu'))
    extended_model.add(keras_layers.Dense(number_of_classes, activation='softmax'))

    extended_model.compile(
        optimizer=tfa.optimizers.LAMB(learning_rate=0.001, weight_decay_rate=0.0001),  # weight decay
        loss=losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return extended_model


"""
Get model from tensorflow addons
@:param type_of_model defines the model to get

Available models: ResNeXt101
@:returns pretrained model
"""


def get_model(type_of_model):
    if type_of_model == 'resnext101':
        # Load ResNeXt101
        ResNeXt101, preprocess_resnext_input = Classifiers.get('resnext101')
        resnext_model = ResNeXt101(include_top=False, weights='imagenet', input_shape=(257, 161, 3))
        return resnext_model, preprocess_resnext_input


"""
Using flow_from_directory() method generalized for validation and training set

@:param image_generator - defines an instance of ImageDataGenerator()
@:param image_directory - directory where folder classes and images are stored. i.e. /whole-net/train
@:param image_height - height of the image below which it is loaded
@:param image_width - width of the image below which it is loaded
@:param batch_size - size of batch 
@:param type - 'training' if we want to select for training ( ratio is defined in image_generator ) 
             - 'validation' if we want to select for validation ( ratio is defined in image_generator )

@:returns DirectoryIterator() (see docs: ImageDataGenerator() -> flow_from_directory)
"""


def get_from_directory(image_generator, image_directory, image_height, image_width, batch_size, type):
    return image_generator.flow_from_directory(
        directory=image_directory,
        target_size=(image_height, image_width),
        subset=type,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=100
    )


"""
Using flow_from_directory() method generalized for testing set

@:param image_generator - defines an instance of ImageDataGenerator()
@:param image_directory - directory where folder classes and images are stored. i.e. /whole-net/test
@:param image_height - height of the image below which it is loaded
@:param image_width - width of the image below which it is loaded
@:param batch_size - size of batch 

@:returns DirectoryIterator() (see docs: ImageDataGenerator() -> flow_from_directory)
"""


def get_from_directory_test(image_generator, image_directory, image_height, image_width, batch_size):
    return image_generator.flow_from_directory(
        directory=image_directory,
        target_size=(image_height, image_width),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )


"""
Saves figure of training history (accuracy, validation loss)
"""


def save_summarize(training, label, val_label):
    plt.plot(training.history[label])
    plt.plot(training.history[val_label])
    plt.title(label)
    plt.ylabel(label)
    plt.xlabel('epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig("results/" + current_time + "/" + label)
    plt.clf()


"""
Plotting ROC curve
"""


def roc_curve_calc(y_score, y_test):
    n_classes = 164
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print(y_test.shape)
    print(y_score.shape)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    new_roc_auc = dict()
    sorted_auc = dict(sorted(roc_auc.items(), key=lambda x: x[1], reverse=True))

    highest_auc = next(iter(sorted_auc.items()))
    new_roc_auc[str(highest_auc[0])] = roc_auc[highest_auc[0]]

    middle_auc = list(sorted_auc.keys())[n_classes // 2]
    new_roc_auc[str(middle_auc)] = roc_auc[middle_auc]

    lowest_auc = list(sorted_auc.keys())[-1]
    new_roc_auc[str(lowest_auc)] = roc_auc[lowest_auc]

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-priemer ROC krivka (AUC = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-priemer ROC krivka (AUC = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    lw = 2
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(new_roc_auc, colors):
        plt.plot(
            fpr[int(i)],
            tpr[int(i)],
            color=color,
            lw=lw,
            label="ROC krivka subjektu {0} (AUC = {1:0.2f})".format(i, roc_auc[int(i)]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Miera falošnej pozitivity")
    plt.ylabel("Miera skutočnej pozitivity")
    plt.title("ROC krivka")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig("results/" + current_time + "/roc")


def get_test_set(test_data, img_height, img_width, batch_size):
    image_generator_test = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255.,
        preprocessing_function=normalize_images
    )

    test_data_from_directory = get_from_directory_test(
        image_generator=image_generator_test,
        image_directory=test_data,
        image_height=img_height,
        image_width=img_width,
        batch_size=batch_size
    )

    return test_data_from_directory


def get_sets(batch_size, img_height, img_width, validation_split, train_data, test_data):
    image_generator_train = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255.,
        validation_split=validation_split,
        preprocessing_function=augment_images
    )

    image_generator_validation = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255.,
        validation_split=validation_split,
        preprocessing_function=normalize_images
    )

    image_generator_test = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255.,
        preprocessing_function=normalize_images
    )

    train_data_from_directory = get_from_directory(
        image_generator=image_generator_train,
        image_directory=train_data,
        image_height=img_height,
        image_width=img_width,
        batch_size=batch_size,
        type="training"
    )

    validation_data_from_directory = get_from_directory(
        image_generator=image_generator_validation,
        image_directory=train_data,
        image_height=img_height,
        image_width=img_width,
        batch_size=batch_size,
        type="validation"
    )

    test_data_from_directory = get_from_directory_test(
        image_generator=image_generator_test,
        image_directory=test_data,
        image_height=img_height,
        image_width=img_width,
        batch_size=batch_size
    )

    return train_data_from_directory, validation_data_from_directory, test_data_from_directory


def training_process(batch_size, img_height, img_width, validation_split, epochs, dataset, to_train, to_test):
    train_data, test_data = get_test_train_by_dataset(dataset, to_train, to_test)

    train, validation, test = get_sets(
        batch_size=batch_size,
        img_height=img_height,
        img_width=img_width,
        validation_split=validation_split,
        train_data=train_data,
        test_data=test_data
    )

    model, preprocess_input = get_model('resnext101')

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    n_classes = np.unique(train.classes).size
    training_model = extend_model_by_head(model, n_classes)

    STEP_SIZE_TRAIN = (train.n // train.batch_size) + 1
    STEP_SIZE_VALID = (validation.n // validation.batch_size) + 1
    STEP_SIZE_TEST = (test.n // test.batch_size) + 1
    print("learning rate before training: ", round(training_model.optimizer.lr.numpy(), 5))

    log_dir = "results/" + current_time + "/tensorboard"

    # callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)
    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduled_learning_rate)

    training = training_model.fit(
        train,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=validation,
        validation_steps=STEP_SIZE_VALID,
        epochs=epochs,
        callbacks=[tensorboard_callback, checkpoint_callback, learning_rate_callback, early_stopping_callback]
    )
    #
    print("learning rate after training: ", round(training_model.optimizer.lr.numpy(), 5))

    training_model.save("results/" + current_time + "/saved-model-both_trained")

    save_summarize(training, 'accuracy', 'val_accuracy')
    save_summarize(training, 'loss', 'val_loss')

    test.reset()
    pred = training_model.predict(test, steps=STEP_SIZE_TEST)
    y_true = test.classes
    true_classes = np.unique(y_true)
    y = label_binarize(y_true, classes=true_classes)

    roc_curve_calc(pred, y)


def evaluate_model(model, batch_size, img_height, img_width, dataset, to_train, to_test):

    train_data, test_data = get_test_train_by_dataset(dataset, to_train, to_test)
    training_model = tf.keras.models.load_model(model)

    test = get_test_set(test_data, img_height, img_width, batch_size)

    STEP_SIZE_TEST = (test.n // test.batch_size) + 1

    test.reset()
    pred = training_model.predict(test, steps=STEP_SIZE_TEST)
    y_true = test.classes
    true_classes = np.unique(y_true)
    y = label_binarize(y_true, classes=true_classes)

    roc_curve_calc(pred, y)


def get_test_train_by_dataset(dataset, to_train, to_test):
    train_data = ''
    test_data = ''

    if "splitted-data" in dataset:
        if to_train == 'right' and to_test == 'left':
            train_data = dataset + "/right_train"
            test_data = dataset + "/left_test"
        elif to_train == 'right' and to_test == 'right':
            train_data = dataset + "/right_train"
            test_data = dataset + "/right_test"
        elif to_train == 'left' and to_test == 'right':
            train_data = dataset + "/left_train"
            test_data = dataset + "/right_test"
        elif to_train == 'left' and to_test == 'left':
            train_data = dataset + "/left_train"
            test_data = dataset + "/left_test"
    else:
        train_data = dataset + "/train"
        test_data = dataset + "/test"

    return train_data, test_data


def main_thread(args):
    # configurable parameters

    batch_size = 50
    image_height = 257
    image_width = 161
    validation_split = 0.2
    epochs = 150

    with open('conf.yml') as yaml_conf:
        dictionary_yaml_configuration = yaml.safe_load(yaml_conf)

    for key, value in dictionary_yaml_configuration.items():
        # generator
        if key == 'generator':
            for inner_key, inner_value in value.items():
                if inner_key == 'batch_size':
                    batch_size = inner_value
                elif inner_key == 'image_height':
                    image_height = inner_value
                elif inner_key == 'image_width':
                    image_width = inner_value
                elif inner_key == 'validation_split':
                    validation_split = inner_value
        # training
        if key == 'training':
            for inner_key, inner_value in value.items():
                if inner_key == 'epochs':
                    epochs = inner_value

    # ------------ switchers --------------

    # define switchers

    parser = argparse.ArgumentParser(description="Ear recognition tool",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--split-whole",
                        help="<path-to-dataset> split sorted dataset into train and test folder (at the end folders contain both sides of ears)",
                        nargs=1)
    parser.add_argument("--split-left-right",
                        help="<path-to-dataset> split sorted dataset into 4 folders (left_train, left_test, right_train, right_test)",
                        nargs=1)
    parser.add_argument("--training",
                        help="<path-to-dataset> starts training process (note: dataset must be available in project folder)", nargs=1)
    parser.add_argument("--preprocess-images",
                        help="<path-to-dataset> preprocess images to 257 x 161 pixels with padding of average color from ImageNet",
                        nargs=1)
    parser.add_argument("--to-train",
                        help="Define if train by left or right ear. Use argument along with --to-test. Values are [left|right] (note: have effect if splitted-data dataset if available in project folder)")
    parser.add_argument("--to-test",
                        help="Define if test by left or right ear. Use argument along with --to-train. Values are [left|right] (note: have effect if splitted-data dataset if available in project folder)")
    parser.add_argument("--evaluate", help="<path-to-model> Evaluate saved model and plot ROC curves. Use argument --to-test to specify which side you want to test", nargs=1)

    arguments = parser.parse_args()
    config = vars(arguments)

    # ----------------------

    training = False
    training_dataset = ''
    to_train = ''
    to_test = ''
    saved_model = ''
    evaluate = False

    try:
        opts, args = getopt.getopt(args, 'h',
                                   ["split-whole=", "split-left-right=", "training=", 'preprocess-images=', 'to-train=',
                                    'to-test='])
    except getopt.GetoptError:
        print(config)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(config)
            sys.exit()
        elif opt == "--split-whole":
            split_whole_net_training(arg)
            sys.exit()
        elif opt == "--split-left-right":
            split_train_test_dataset(arg)
            sys.exit()
        elif opt == "--training":
            training = True
            training_dataset = arg
        elif opt == "--preprocess-images":
            walk_through_images_and_resize(arg)
            sys.exit()
        elif opt == '--evaluate':
            saved_model = arg
            evaluate = True
        elif opt == "--to-train":
            to_train = arg
        elif opt == "--to-test":
            to_test = arg

    if training:
        training_process(
            batch_size=batch_size,
            img_height=image_height,
            img_width=image_width,
            validation_split=validation_split,
            epochs=epochs,
            dataset=training_dataset,
            to_train=to_train,
            to_test=to_test
        )

    if evaluate:
        evaluate_model(
            model=saved_model,
            to_test=to_test
        )


if __name__ == '__main__':
    main_thread(sys.argv[1:])

