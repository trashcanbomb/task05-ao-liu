# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import monai
import os
from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from models_new import UNet3D
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
import torch
import warnings
# 禁用所有警告
warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')
directory = "../dataset_news"
os.makedirs(directory, exist_ok=True)
root_dir = directory
print(root_dir)

set_determinism(seed=0)


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 2)
            result.append(d[key] == 1)
            d[key] = torch.stack(result, axis=0).float()
        return d


val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

val_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task05_Prostate",
    transform=val_transform,
    section="validation",
    download=False,
    cache_rate=0.0,
    num_workers=0,
)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

max_epochs = 500
val_interval = 1
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")

model = UNet3D().to(device)
model.load_state_dict(torch.load( "best_metric_model_new_v1.pth"))


dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(160, 160, 48),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


torch.backends.cudnn.benchmark = True

epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []


model.eval()
with torch.no_grad():
    for val_data in val_loader:
        val_inputs, val_labels = (
            val_data["image"].to(device),
            val_data["label"].to(device),
        )
        val_outputs = inference(val_inputs)
        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

        import cv2
        os.makedirs("output",exist_ok=True)
        for t in range(val_data["image"][0].shape[3]):
            cv2.imwrite("output/predict_0_" + str(t) + '.jpg', (val_outputs[0][0][:, :, t] * 255).cpu().numpy())
            cv2.imwrite("output/predict_1_" + str(t) + '.jpg', (val_outputs[0][1][:, :, t] * 255).cpu().numpy())
            cv2.imwrite("output/image_0_" + str(t) + '.jpg', (val_inputs[0][0][:, :, t] * 255).cpu().numpy())
            cv2.imwrite("output/image_1_" + str(t) + '.jpg', (val_inputs[0][0][:, :, t] * 255).cpu().numpy())
            cv2.imwrite("output/label_0_" + str(t) + '.jpg', (val_labels[0][0][:, :, t] * 255).cpu().numpy())
            cv2.imwrite("output/label_1_" + str(t) + '.jpg', (val_labels[0][0][:, :, t] * 255).cpu().numpy())
        exit()
