import torch
from PIL import Image
import cv2
import numpy as np
import random
import os
import torchvision.transforms as transform
from geogeniustools.s3 import S3
from object_extraction.encoding.models import get_segmentation_model
from geogeniustools.images.obs_image import OBSImage
from geogeniustools.rda.io import TiffFactory
torch.backends.cudnn.benchmark = True
sub_h = 512
sub_w = 512
step = 256
pad = 20
num_classes = 6
BATCH_SIZE = 4


infer_transformation = transform.Compose([
    transform.Resize((512,512)),
    transform.ToTensor(),
    transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

use_cuda = torch.cuda.is_available()


def split_data(large_img):
    res_list = []
    tmp_list = []
    offset_list = []
    h, w, c = large_img.shape
    for i in range(0, h, step):
        for j in range(0, w, step):
            if i + sub_h > h:
                offset_h = h - sub_h
            else:
                offset_h = i
            if j + sub_w > w:
                offset_w = w - sub_w
            else:
                offset_w = j
            new_img = large_img[offset_h : (offset_h + sub_h),
                                offset_w : (offset_w + sub_w), :]
            new_img = infer_transformation(Image.fromarray(new_img))
            tmp_list.append(new_img)
            offset_list.append([offset_h, offset_w])
    num_list = list(range(len(tmp_list)))
    random.shuffle(num_list)
    tmp_list = [tmp_list[idx] for idx in num_list]
    offset_list = [offset_list[idx] for idx in num_list]
    new_batch = torch.zeros((BATCH_SIZE, c, sub_h, sub_w))
    cnt = 0
    for new_img in tmp_list:
        if cnt == BATCH_SIZE:
            res_list.append(new_batch)
            new_batch = torch.zeros((BATCH_SIZE, c, sub_h, sub_w))
            cnt = 0
        new_batch[cnt, :, :, :] = new_img[:, :, :]
        cnt += 1
    if cnt != 0:
        res_list.append(new_batch)
    return res_list, offset_list

def predict_process_pv(model_path, img_path, img_obs_path, output_path):
    local_dir = os.path.abspath("model_file")
    client = S3()
    client.download(obs_path=model_path, local_dir=local_dir)
    file_name = model_path.split('/')[len(model_path.split('/')) - 1]

    model = get_segmentation_model('danet', dataset='norm', backbone='resnet50', aux=False, se_loss=False,
                                   norm_layer=torch.nn.BatchNorm2d, base_size=512, crop_size=512, multi_grid=True,
                                   multi_dilation=(4, 8, 16))

    try:
        if use_cuda:
            model = model.to("cuda:0")
            checkpoint = torch.load(os.path.join(local_dir, file_name))
        else:
            checkpoint = torch.load(os.path.join(local_dir, file_name), map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    except Exception as e:
        print("error when load model file: ", e)
        exit(1)
    large_img = cv2.imread(img_path, 4)
    large_img = cv2.copyMakeBorder(large_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    large_img = cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB)
    data_list, offset_list = split_data(large_img)
    large_bin_pred = np.zeros(large_img.shape[:2])
    print (len(data_list))
    for idx, input_data in enumerate(data_list):
        if use_cuda:
            input_data = input_data.cuda(0)
        output = model.evaluate(input_data)
        pre = torch.max(output, 1)[1].cpu().numpy() + 1
        for sub_idx in range(BATCH_SIZE):
            if (idx * BATCH_SIZE + sub_idx) >= len(offset_list):
                break
            offset_h = offset_list[idx * BATCH_SIZE + sub_idx][0]
            offset_w = offset_list[idx * BATCH_SIZE + sub_idx][1]
            large_bin_pred[offset_h + pad : offset_h + sub_h - pad,
                        offset_w + pad : offset_w + sub_w - pad] = pre[sub_idx][pad:-pad, pad:-pad]
    result = large_bin_pred[pad:-pad, pad:-pad]
    origin_img = OBSImage(img_obs_path)
    mask_img = np.max(origin_img.read().transpose(1, 2, 0)[:, :, :3], axis=2)
    mask_img[mask_img > 0] = 1
    result = result * mask_img
    factory = TiffFactory()
    factory.generate_tiff_from_array(origin_img.get_image_meta(), result, output_path)

