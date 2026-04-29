import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from sklearn.utils import shuffle

from Evaluation import net_evaluation
from Unet import unet_opt
from tf_data import tf_dataset
from tta import tta_model


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def Model_TransUnetPlusPlus(Image, GT, sol=None):
    if sol is None:
        sol = [5, 5, 1]
    tf.random.set_seed(42)
    np.random.seed(42)

    ## Create files folder
    try:
        os.mkdir("files")
    except:
        pass

    ## Training
    train_image_paths = Image
    train_mask_paths = sorted(glob(os.path.join(GT, "mask*.png")))

    ## Shuffling
    train_image_paths, train_mask_paths = shuffling(train_image_paths, train_mask_paths)

    # ## Validation
    # valid_image_paths = sorted(glob(os.path.join(valid_path, "image*.png")))
    # valid_mask_paths = sorted(glob(os.path.join(valid_path, "mask*.png")))

    ## Parameters
    image_size = 256
    batch_size = 16
    # lr = sol[0]
    epochs = int(sol[1])

    train_dataset = tf_dataset(train_image_paths, train_mask_paths)
    # valid_dataset = tf_dataset(valid_image_paths, valid_mask_paths)

    try:
        arch = unet_opt(input_size=image_size)
        model = arch.build_model()
        model = tf.distribute.MirroredStrategy(model, 4, cpu_merge=False)
        print("Training using multiple GPUs..")
    except:
        arch = unet_opt(input_size=image_size)
        model = arch.build_model()
        print("Training using single GPU or CPU..")

    optimizer = Nadam(learning_rate=0.001)
    # metrics = [dice_coef, MeanIoU(num_classes=2), Recall(), Precision()]
    metrics = [MeanIoU(num_classes=2), Recall(), Precision()]
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
    model.summary()

    callbacks = [
        # ModelCheckpoint(model_path),
        #         ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
        # schedule
    ]

    train_steps = (len(train_image_paths) // batch_size)
    # valid_steps = (len(valid_image_paths) // batch_size)

    if len(train_image_paths) % batch_size != 0:
        train_steps += 1

    # if len(valid_image_paths) % batch_size != 0:
    #     valid_steps += 1

    model.fit(train_dataset,
              epochs=epochs,
              validation_data=Image,
              steps_per_epoch=int(sol[2]),
              validation_steps=0.3,
              callbacks=callbacks,
              shuffle=False)

    Seg_Images = []
    Target = []
    for i in range(len(Image)):
        image = Image[i]
        mask = GT[i]
        pred_mask = tta_model(model, image)
        pred_mask = pred_mask.squeeze()
        Seg_Images.append(pred_mask)
        Target.append(mask)
    Eval = net_evaluation(Seg_Images, Image)

    # model.save(model_name)
    return Eval, Seg_Images
