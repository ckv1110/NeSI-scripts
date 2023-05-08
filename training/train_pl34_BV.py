from typing import Optional, List
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import albumentations as A
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from segmentation_models_pytorch import utils
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint as mc

# Variables setup ==========================================
VERSION = "15"
# MULTILABEL_MODE: str = "multilabel"
ARCHITECTURE = "UnetPlusPlus"
ENCODER = "resnet34"
ENCODER_WEIGHTS = 'imagenet'
# ACTIVATION = 'softmax2d' # Could be None to obtain raw logits, 'sigmoid' for binary or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# Modify directory vars here ==================
os.chdir("/var/inputdata")
dataset = "BV_dataset_16-01-23.csv"
fdf = pd.read_csv(dataset)
# pth_s_name = f'S:/PhD_work/coding/from_NeSI/scripts and py/NeSI-scripts{ENCODER}_UNetPP_best_model.pth'

# Set logging dir =======================
tb_logger = pl_loggers.TensorBoardLogger(save_dir='tb_logs/')
csv_logger = pl_loggers.CSVLogger(save_dir='csv_logs/')

# Create dataset that Dataloader can read ========================
class TumourDataset(Dataset):
    # Modify classes here too
    CLASSES = ['Background', 'Hyperplastic vessels', 'Microvascular proliferation (MVP)', 'Vessels (not MVP)']

    def __init__(
            self,
            df,
            classes=None,
            transforms=None,
            preprocessing=None,
    ):
        self.df = df
        self.class_values = [self.CLASSES.index(cls) for cls in classes]
        self.transforms = transforms
        self.preprocessing = preprocessing

    # Grab image and mask from dataframe (df)
    def __getitem__(self, idx):
        image_name = self.df.iloc[idx, 0]
        mask_name = self.df.iloc[idx, 1]
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
        mask = mask[:, :, 0]

        # Extract classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        # mask = mask.astype('float')
        # print(mask.shape)

        # Apply transforms and augmentations
        if self.transforms:
            sample = self.transforms(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.df)

# Class to create model, with losses and optimizers specified here ===========
class TumourSegModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, **kwargs):
        super().__init__()
        # Model creation is done here. Must specify architecture (arch), encoder, number of channels in images and number of classes =======
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs
        )

        # Set up loss here.
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
            #smp.losses.TverskyLoss(mode='multiclass', from_logits=True)

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]
        mask = batch[1]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        # gt = mask
        # gt_mask = torch.from_numpy(self.combine_masks(gt.cpu()))
        # print(gt_mask.shape, logits_mask.shape)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask.long())

        # Let's compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        # m = nn.Softmax(dim=1)
        # prob_mask = m(logits_mask)
        # pred_mask = torch.argmax(prob_mask, dim=1).unsqueeze(1)

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multilabel", num_classes=len(TumourDataset.CLASSES))

        self.log(f'{stage}_loss', loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metrics
        loss = torch.stack([x["loss"] for x in outputs])
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        # self.log(f'{stage}_loss_end_epoch', loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log(f"{stage}_per_image_iou", per_image_iou, on_epoch=True, sync_dist=True, prog_bar=False, logger=True)
        self.log(f"{stage}_dataset_iou", dataset_iou, on_epoch=True, sync_dist=True, prog_bar=False, logger=True)

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        # self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00001)

# Function to split the whole dataset into training and validation splits ===========
# Does not yield a testing split ("validation" in the writing of our milestone), since we are using the Ivy GAP dataset
def split_data(fdf):
    y = fdf.mask_path
    X = fdf.img_path
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    trdf = pd.DataFrame()
    trdf.insert(loc=0, column='img_path', value=X_train)
    trdf.insert(loc=1, column='mask_path', value=y_train)
    tedf = pd.DataFrame()
    tedf.insert(loc=0, column='img_path', value=X_test)
    tedf.insert(loc=1, column='mask_path', value=y_test)
    return trdf, tedf

# Helper function to visualize things ==============================
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap=plt.get_cmap('gray'),vmin=0,vmax=10)
    plt.show()

# Set up albumentation transformations for data augmentation with training dataset =======
def train_transforms():
    t_transforms = [
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=4),
        A.Resize(512, 512),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.MedianBlur(blur_limit=5, always_apply=False, p=0.5),
        # A.Lambda(image=to_tensor, mask=to_tensor)
    ]
    return A.Compose(t_transforms)

# Set up albumentation transformations for data augmentation with validation dataset =======
def val_transforms():
    v_transforms = [A.PadIfNeeded(min_height=512, min_width=512, border_mode=4), A.Resize(512, 512),
                    # A.Lambda(image=to_tensor, mask=to_tensor)
                    ]
    return A.Compose(v_transforms)

# To tensor function =============================================
def to_tensor(x, **kwargs):
    return x.transpose(2,0,1).astype('float32')

# Set up preprocessing transforms specific to ENCODER and ENCODER_WEIGHTS ============
def preprocessing_transforms(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

if __name__ == '__main__':
    # Load training and testing splits from csv using above funciton =====================
    train, val = split_data(fdf)

    # Check if GPU is available ===================================
    avail = torch.cuda.is_available()
    devCnt = torch.cuda.device_count()
    devName = torch.cuda.get_device_name(0)
    print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))

    # Get preprocessing functions from smp based on encoder and encoder weights
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # Set your training and validation datasets ========================
    trainDS = TumourDataset(train, classes=TumourDataset.CLASSES, transforms=train_transforms(), preprocessing=preprocessing_transforms(preprocessing_fn))
    valDS = TumourDataset(val, classes=TumourDataset.CLASSES, transforms=val_transforms(), preprocessing=preprocessing_transforms(preprocessing_fn))
    print("Number of Training Samples: " + str(len(train)) + "\nNumber of Validation Samples: " + str(len(val)))

    # Load datasets (14,3 seems good)
    trainDL = DataLoader(trainDS, batch_size=14, shuffle=True)
    valDL = DataLoader(valDS, batch_size=14, shuffle=False)
    # Sanity check shapes and images in dataloaders
    t_batch = next(iter(trainDL))
    images, labels = t_batch
    print(images.shape, labels.shape, type(images), type(labels), images.dtype, labels.dtype)

    # Set up models with variables set in the begining. in_channels refer to channels of input image (RGB = 3 channels)
    # out_classes refers to number of classes you are working with (check with teh CLASSES variable that you have set) ==================
    model = TumourSegModel(
        ARCHITECTURE,
        ENCODER,
        ENCODER_WEIGHTS,
        in_channels=3,
        out_classes=len(TumourDataset.CLASSES),
    )

    # Set how often you check your validation metrics ===============
    validation_interval = 1.0

    # Set how checkpoints are saved ===========================
    cp_callback = mc(dirpath='BV_checkpoints/',
                     save_top_k=10,
                     monitor='val_loss_epoch',
                     save_on_train_epoch_end=True,
                     filename=f'{ARCHITECTURE}_{ENCODER}_{VERSION}_BV_diceloss' + '_{epoch:02d}_{val_loss_epoch:.3f}'
                     )

    # Sets up trainer by initializing all GPUs available and define max epoxh =============
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        max_epochs=100,
        val_check_interval=validation_interval,
        logger=[tb_logger, csv_logger],
        callbacks=[cp_callback]
    )

    # # Start the training ==========================================
    trainer.fit(
        model,
        train_dataloaders=trainDL,
        val_dataloaders=valDL,
    )

    # Load checkpoint ===============================
    # trained_model = TumourSegModel.load_from_checkpoint(
    #     checkpoint_path='checkpoints/UnetPlusPlus_resnet34_9_focalloss_epoch=99_val_loss_epoch=0.115.ckpt',
    #     arch=ARCHITECTURE,
    #     encoder_name=ENCODER,
    #     encoder_weights=ENCODER_WEIGHTS,
    #     in_channels=3,
    #     out_classes=len(TumourDataset.CLASSES),
    #     )