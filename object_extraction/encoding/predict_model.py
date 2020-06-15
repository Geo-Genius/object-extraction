import sys
sys.path.append("..")
from geogeniustools.eolearn.geogenius_areas import PixelRangeSplitter
from geogeniustools.eolearn.geogenius_set import GeogeniusPatchSet
from geogeniustools.images.mosaic_image import MosaicImage
from eolearn.core import FeatureType, AddFeature
from geogeniustools.images.obs_image import OBSImage
from geogeniustools.s3 import S3
import torch
from PIL import Image
import torchvision.transforms as transform
from object_extraction.encoding.models import get_segmentation_model
from geogeniustools.rda.io import TiffFactory
import os
import numpy as np
import random
import cv2

torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
sub_h = 512
sub_w = 512
step = 256
pad = 20
BATCH_SIZE = 4


infer_transformation = transform.Compose([
    transform.Resize((512, 512)),
    transform.ToTensor(),
    transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])


def predict_process(output_path, model_path, cat_ids=None, paths=None, aoi=None):
    client = S3()
    bucket = client.info.get("bucket")
    padded_temp_tiff = "obs://%s/temp.tif" % bucket
    try:
        print("start check environment variable...")
        obs_env_check()
        print("start load model...")
        model = load_model(model_path)
        print("start load data...")
        cat_ids = split2list(cat_ids)
        paths = split2list(paths)
        if aoi is not None:
            aoi = split2list(aoi)
            aoi = [float(i) for i in aoi]
        patch_set = load_data(cat_ids=cat_ids, paths=paths, padded_temp_tiff=padded_temp_tiff, aoi=aoi)
        print("start predict...")
        predict_model(model=model, patch_set=patch_set)
        print("write to obs...")
        patch_set.save_to_obstiff(output_path, (FeatureType.DATA, 'MASK'), no_data_value=0, padding=20)
        print("predict model done")
    finally:
        client.delete(padded_temp_tiff)


def load_data(cat_ids, paths, padded_temp_tiff, aoi):
    """
    read tiff from OBS and split into GeogeniusPatchSet
    """
    img = MosaicImage(cat_ids=cat_ids, paths=paths)
    factory = TiffFactory()
    factory.generate_padded_tiff(img.get_image_meta(), img.read(), pad_width=((0, 0), (20, 20), (20, 20)),
                                 obs_path=padded_temp_tiff)
    # img.padtiff(obs_path=padded_temp_tiff, pad_width=((20, 20), (20, 20), (0, 0)))
    pad_img = OBSImage(padded_temp_tiff)
    if aoi is not None and len(aoi) == 4:
        print("start calculate aio...")
        pad_img = pad_img.aoi(bbox=aoi)
    bbox_splitter = PixelRangeSplitter(pad_img.shape[1:], (sub_h, sub_w), (step, step))
    patch_set = GeogeniusPatchSet(pad_img, bbox_splitter)
    return patch_set


def load_model(model_path):
    """
    load model with cpu or gpu mode
    """
    local_dir = os.path.abspath("model_file")
    client = S3()
    client.download(obs_path=model_path, local_dir=local_dir)
    file_name = model_path.split('/')[len(model_path.split('/')) - 1]
    model = get_segmentation_model(
        'danet',
        dataset='norm',
        backbone='resnet50',
        aux=False,
        se_loss=False,
        norm_layer=torch.nn.BatchNorm2d,
        base_size=512,
        crop_size=512,
        multi_grid=True,
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
    return model


def split_data(patch_set):
    shape = patch_set.shape
    res_list = []
    tmp_list = []
    patch_list = []
    mask_list = []
    for h in range(shape[0]):
        for w in range(shape[1]):
            patch = patch_set.patch_index[h][w]
            img = patch.data['BANDS'][:, :, :, 0:3].squeeze()
            mask_img = np.max(img, axis=2)
            mask_img[mask_img > 0] = 1
            new_img = infer_transformation(Image.fromarray(img))
            tmp_list.append(new_img)
            patch_list.append(patch)
            mask_list.append(mask_img)
    num_list = list(range(len(tmp_list)))
    random.shuffle(num_list)
    tmp_list = [tmp_list[idx] for idx in num_list]
    patch_list = [patch_list[idx] for idx in num_list]
    mask_list = [mask_list[idx] for idx in num_list]
    new_batch = torch.zeros((BATCH_SIZE, 3, sub_h, sub_w))
    cnt = 0
    for new_img in tmp_list:
        if cnt == BATCH_SIZE:
            res_list.append(new_batch)
            new_batch = torch.zeros((BATCH_SIZE, 3, sub_h, sub_w))
            cnt = 0
        new_batch[cnt, :, :, :] = new_img[:, :, :]
        cnt += 1
    if cnt != 0:
        res_list.append(new_batch)
    return res_list, patch_list, mask_list


def predict_model(model, patch_set):
    """
    predict model and return a result list
    """
    data_list, patch_list, mask_list = split_data(patch_set)
    for idx, input_data in enumerate(data_list):
        if use_cuda:
            input_data = input_data.cuda(0)
        output = model.evaluate(input_data)
        pre = torch.max(output, 1)[1].cpu().numpy() + 1
        for sub_idx in range(BATCH_SIZE):
            if (idx * BATCH_SIZE + sub_idx) >= len(patch_list):
                break
            patch_idx = idx * BATCH_SIZE + sub_idx
            res = pre[sub_idx] * mask_list[patch_idx]
            res = res.astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            image_dealed = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
            mask_feature = (FeatureType.DATA, 'MASK')
            add_mask_feature = AddFeature(mask_feature)
            add_mask_feature.execute(patch_list[patch_idx], image_dealed[np.newaxis, :, :, np.newaxis])


def obs_env_check():
    aws_env_list = ["ACCESS_KEY", "SECRET_KEY"]
    for env in aws_env_list:
        env_check(env)


def env_check(key):
    value = os.environ.get(key)
    if value is None:
        sys.stderr.write("environment variable for %s is not set.\n" % key)
        exit(1)


def split2list(str):
    if str is None:
        return None
    else:
        try:
            return str.split(",")
        except Exception as e:
            print("error when analyse parameters: ", e)
            sys.stderr.write(e)
            exit(1)



if __name__ == '__main__':
    cat_ids = None
    # 推理文件obs存储位置
    paths = 'obs://geogenius-public-bucket/geogenius/1577517737080/top_potsdam_2_11_RGBIR.tif'
    label_paths = 'obs://geogenius-public-bucket/geogenius/1577523025371/top_potsdam_2_10_label.tif'
    model_file = 'obs://geogenius-bucket/AI-model/object-extraction/model_file/model_best.pth.tar'
    output_path = 'obs://geo-test7f2953a3/test/result.tif'
    aoi = None
    # 推理结果本地存储位置
    # output_path = '/tmp/test.pth.tar'
    predict_process(output_path=output_path, paths=paths, model_path=model_file)






