#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import shutil
import traceback

from configs import paths_config, global_config
from utils import align_data
from scripts import run_pti
import os
import torch
import argparse

parser = argparse.ArgumentParser(description='大语言模型后端项目')
parser.add_argument('--gpu', type=str, help='gpuid', default='0')
parser.add_argument('--image', type=str, help='image_name', default='raw')
args = parser.parse_args()

global_config.cuda_visible_devices = args.gpu

# 待处理目录
paths_config.checkpoints_dir = '/data/home/yaokj5/dl/apps/DragGAN/PTI'
image_original = os.path.join(paths_config.checkpoints_dir, 'image_original')
paths_config.input_data_path = os.path.join(paths_config.checkpoints_dir, 'image_processed')
paths_config.stylegan2_ada_ffhq = os.path.join(paths_config.checkpoints_dir, 'pretrained_models', 'ffhq.pkl')
paths_config.style_clip_pretrained_mappers = os.path.join(paths_config.checkpoints_dir, 'pretrained_models')

os.makedirs(paths_config.input_data_path, exist_ok=True)
shutil.rmtree(paths_config.input_data_path, ignore_errors=True)
os.makedirs(image_original, exist_ok=True)

try:
    align_data.pre_process_images(image_original)
except Exception as e:
    traceback.print_exc()
    raise e
else:
    print('图片预处理成功')

"""
使用 PTI 进行 GAN 反演

反演是指将一个图像映射到生成模型的潜空间中，然后通过调整潜空间向量来修改图像的外观。通过这种方式，可以实现对图像的各种编辑操作，
例如改变姿势、修改外貌特征或添加不同的风格。通过编辑潜空间，可以实现对图像的高级编辑，同时保持图像的真实性和准确性。
"""
use_multi_id_training = False
model_id = run_pti.run_PTI(use_multi_id_training=use_multi_id_training)

"""
PTI 反演后的文件不是 DragGAN 可识别的模型文件格式

保存为 DragGAN 可识别的模型文件
#code from : https://github.com/danielroich/PTI/issues/26 , plus little bit modification
"""


def load_generators(model_id, image_name):
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].cuda()

    with open(f'{paths_config.checkpoints_dir}/model_{model_id}_{image_name}.pt', 'rb') as f_new:
        new_G = torch.load(f_new).cuda()

    return old_G, new_G


def export_updated_pickle(new_G, model_id):
    print("Exporting large updated pickle based off new generator and ffhq.pkl")
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        d = pickle.load(f)
        old_G = d['G_ema'].cuda()
        old_D = d['D'].eval().requires_grad_(False).cpu()

    tmp = {}
    tmp['G'] = old_G.eval().requires_grad_(False).cpu()
    tmp['G_ema'] = new_G.eval().requires_grad_(False).cpu()
    tmp['D'] = old_D
    tmp['training_set_kwargs'] = None
    tmp['augment_pipe'] = None

    with open(f'{paths_config.checkpoints_dir}/stylegan2_custom_512_pytorch.pkl', 'wb') as f:
        pickle.dump(tmp, f)


# checkpoints 目录下 pt 文件名的一部分
image_name = args.image
generator_type = paths_config.multi_id_model_type if use_multi_id_training else image_name
old_G, new_G = load_generators(model_id, generator_type)
export_updated_pickle(new_G, model_id)
