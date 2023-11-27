import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/segmentation_models_pytorch/')
import segmentation_models_pytorch as smp
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DATA_DIR = './carla_data'

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset





class Dataset(BaseDataset):
    CLASSES = ['unlabelled', 'building', 'fence', 'other', 'pedestrian', 
               'pole', 'roadline', 'road', 'sidewalk', 'vegetation', 
               'car', 'wall', 'trafficsign'] 
    CLASSES_COLOR = {
        0	:	[ 0, 0, 0],
        1	:	[ 70, 70, 70],
        2	:   [190, 153, 153],
        3	:	[250, 170, 160],
        4	:   [220, 20, 60],
        5	:	[153, 153, 153],
        6	: 	[157, 234, 50],
        7	:	[128, 64, 128],
        8	:   [244, 35, 232],
        9	:	[107, 142, 35],
        10	:   [ 0, 0, 142],
        11	:	[102, 102, 156],
        12	:	[220, 220, 0]
    }
    # CLASSES = ['unlabeled'    ,              
    #     'dynamic'      ,        
    #     'ground'       ,        
    #     'road'         ,        
    #     'sidewalk'     ,        
    #     'parking'      ,        
    #     'rail track'   ,        
    #     'building'     ,        
    #     'wall'         ,        
    #     'fence'        ,        
    #     'guard rail'   ,        
    #     'bridge'       ,        
    #     'tunnel'       ,        
    #     'pole'         ,        
    #     'polegroup'    ,        
    #     'traffic light',        
    #     'traffic sign' ,        
    #     'vegetation'   ,        
    #     'terrain'      ,        
    #     'sky'          ,        
    #     'person'       ,        
    #     'rider'        ,        
    #     'car'          ,        
    #     'truck'        ,        
    #     'bus'          ,        
    #     'caravan'      ,        
    #     'trailer'      ,        
    #     'train'        ,        
    #     'motorcycle'   ,        
    #     'bicycle'      ,        
    #     'license plate'
    # ]
    # CLASSES_COLOR = {
    # 0   : [  0,  0,  0],
    # 1   : [111, 74,  0],
    # 2   : [ 81,  0, 81],
    # 3   : [128, 64,128],
    # 4   : [244, 35,232],
    # 5   : [250,170,160],
    # 6   : [230,150,140],
    # 7   : [ 70, 70, 70],
    # 8   : [102,102,156],
    # 19  : [190,153,153],
    # 10  : [180,165,180],
    # 11  : [150,100,100],
    # 12  : [150,120, 90],
    # 13  : [153,153,153],
    # 14  : [153,153,153],
    # 15  : [250,170, 30],
    # 16  : [220,220,  0],
    # 17  : [107,142, 35],
    # 18  : [152,251,152],
    # 19  : [ 70,130,180],
    # 20  : [220, 20, 60],
    # 21  : [255,  0,  0],
    # 22  : [  0,  0,142],
    # 23  : [  0,  0, 70],
    # 24  : [  0, 60,100],
    # 25  : [  0,  0, 90],
    # 26  : [  0,  0,110],
    # 27  : [  0, 80,100],
    # 28  : [  0,  0,230],
    # 29  : [119, 11, 32],
    # 20  : [  0,  0,142]
    # }

    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir,image_id ) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (480, 360))
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (480, 360))

        new_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(0, len(self.CLASSES_COLOR)):
            new_mask[np.where(np.all(mask == self.CLASSES_COLOR[i], axis=-1))] = i
        mask = new_mask
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)


### Augmentations
import albumentations as albu
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)

    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

ENCODER = 'resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
# ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
ACTIVATION = 'softmax2d'
DEVICE = 'cuda'
# CLASSES = ['unlabeled','dynamic','ground','road','sidewalk','parking','rail track','building','wall','fence','guard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','license plate']
CLASSES = ['unlabelled', 'building', 'fence', 'other', 'pedestrian', 
               'pole', 'roadline', 'road', 'sidewalk', 'vegetation', 
               'car', 'wall', 'trafficsign'] 
# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])
# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# train model for 40 epochs
max_score = 0

for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

## Test best saved model
# load best saved checkpoint
best_model = torch.load('./best_model.pth')
