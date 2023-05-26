import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional
import cv2
import sys
import argparse

from segment_anything import sam_model_registry, SamPredictor


def img_to_tensor(img):
    return (torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.).unsqueeze(0)


def tensor_to_img(img):
    return (img[0].data.cpu().numpy().transpose((1, 2, 0)).clip(0, 1) * 255 + 0.5).astype(np.uint8)


def resize(img, long_side=512, keep_ratio=True):
    if keep_ratio:
        h, w = img.shape[:2]
        if h < w:
            new_h = int(long_side * h / w)
            new_w = int(long_side)
        else:
            new_w = int(long_side * w / h)
            new_h = int(long_side)
        return cv2.resize(img, (new_w, new_h))
    else:
        return cv2.resize(img, (long_side, long_side))


def padding(img, factor=32):
    h, w = img.shape[:2]
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    new_img = np.zeros((h + pad_h, w + pad_w, img.shape[2]), dtype=img.dtype)
    new_img[:h, :w, :] = img
    return new_img

# 这个函数的作用是计算一个四维张量（batch size, channel, height, width）中每个通道沿着高度和宽度方向的像素值的平均值和标准差。
# 这个函数通常用于图像风格转换等计算机视觉任务中，用于规范化图像数据以便进行模型训练。在训练神经网络时，规范化数据可以使模型更容易收敛，并提高训练效果。
# 定义一个名为 calc_mean_std 的函数，它接受一个张量feat和一个eps参数。
def calc_mean_std(feat, eps=1e-5):
    # eps 是一个小值，加到方差上，避免除以零。
    size = feat.size()
    # 确保输入张量是四维的（batch size, channel, height, width）。
    assert (len(size) == 4)
    # 获取batch size和channel的数量。
    N, C = size[:2]
    # 计算沿着高度和宽度方向的像素值的方差，加上eps以避免除以零。
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    # 计算标准差，它是方差的平方根。
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    # 计算像素值的平均值。
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    # 返回像素值的平均值和标准差。
    return feat_mean, feat_std


# 对一个四维张量（batch size, channel, height, width）进行规范化，使用输入张量的平均值和标准差进行归一化。
def mean_variance_norm(feat):
    # 获取输入张量的尺寸。
    size = feat.size()
    # 使用 calc_mean_std 函数计算输入张量的平均值和标准差。
    mean, std = calc_mean_std(feat)
    # 对输入张量进行规范化，使用平均值和标准差进行归一化。
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    # 返回规范化后的张量。
    return normalized_feat

# 函数将最后一层之前的所有特征张量进行规范化，并使用双线性插值将它们的尺寸调整为与最后一层特征张量相同。然后，函数对最后一层特征张量进行规范化。
# 最后，函数将所有规范化后的特征张量按通道维度连接起来，并返回连接后的张量。这个函数通常用于计算机视觉任务中的特征提取和图像风格转换等应用。
def get_key(feats, last_layer_idx):
    # 创建一个空列表，用于存储规范化后的特征张量。
    results = []
    # 获取最后一层特征张量的高度和宽度。
    _, _, h, w = feats[last_layer_idx].shape
    # 对最后一层之前的特征张量进行规范化，并使用双线性插值将它们的尺寸调整为与最后一层特征张量相同。
    for i in range(last_layer_idx):
        results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
    # 对最后一层特征张量进行规范化。
    results.append(mean_variance_norm(feats[last_layer_idx]))
    # 按通道维度连接所有规范化后的特征张量。
    return torch.cat(results, dim=1)

# 该函数的作用是实现自适应注意力机制，用于计算内容图像和风格图像之间的注意力权重，并将这些权重应用于内容图像的特征张量和风格图像的特征张量之间的加权平均值和标准差计算。
# 具体来说，该函数接受两个输入：内容图像和风格图像，并使用它们之间的特征张量来计算自适应注意力权重，然后将这些权重应用于风格图像的特征张量，以生成风格化的内容图像的特征张量。

# 用于执行自适应注意力和 AdaIN 操作。它接受两个输入：内容特征向量和风格特征向量。它使用内容特征向量和风格特征向量来计算注意力权重，然后使用注意力权重对内容特征向量进行加权，
# 以生成风格化的特征向量。 最后，它使用 AdaIN 操作将风格化的特征向量与输入的内容特征向量合并，生成风格化的输出特征向量。这个函数通常用于风格迁移网络中，用于将给定的内容图像转换为给定的风格图像。
class AdaAttN(nn.Module):

    # 定义一个名为 __init__ 的方法，用于初始化 AdaAttN 类。
    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        # 调用父类的 __init__ 方法。
        super(AdaAttN, self).__init__()
        # 如果未指定 key_planes，则将其设置为 in_planes。
        if key_planes is None:
            key_planes = in_planes
        # 创建三个卷积层，其中 f 和 g 用于提取特征向量，h 用于生成风格化的特征向量。
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        # 创建一个 Softmax 层，用于计算注意力权重。
        self.sm = nn.Softmax(dim=-1)
        # 设置最大样本继续上一个回答：数量，用于在样本数超过最大样本数量时对样本进行下采样。
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None, content_masks=None, style_masks=None):
        # 如果未提供内容遮罩，则将其设置为空列表。
        if content_masks is None:
            content_masks = []
        # 如果未提供样式遮罩，则将其设置为空列表。
        if style_masks is None:
            style_masks = []
        # 用 f 和 g 提取内容和样式的特征向量。
        F = self.f(content_key)
        G = self.g(style_key)
        # 用 h 生成风格化的特征向量。
        H = self.h(style)
        # 重塑 G 的形状。
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        G = G.view(b, -1, w_g * h_g).contiguous()
        # 将样式遮罩插值到与 G 相同的分辨率，并将其转换为相应的形状。
        style_masks = [nn.functional.interpolate(mask, size=(h_g, w_g), mode='nearest').view(
            b, 1, h_g * w_g).contiguous() for mask in style_masks]
        # 如果样式向量的数量超过最大样本数量，则从中随机选择最大样本数量个样本。
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_masks = [mask[:, :, index] for mask in style_masks]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        content_masks = [nn.functional.interpolate(mask, size=(h, w), mode='nearest').view(
            b, 1, w * h).permute(0, 2, 1).contiguous() for mask in content_masks]
        S = torch.bmm(F, G)
        for content_mask, style_mask in zip(content_masks, style_masks):
            style_mask = 1. - style_mask
            attn_mask = torch.bmm(content_mask, style_mask)
            S = S.masked_fill(attn_mask.bool(), -1e15)
        # S: b, n_c, n_s
        # 计算注意力加权的均值和标准差。
        S = self.sm(S)
        # mean: b, n_c, c
        # 将均值和标准差重塑为与内容特征向量相同的形状。
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        # 对内容进行均值方差归一化，然后添加风格化的均值。
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean


class Transformer(nn.Module):
    def __init__(self, in_planes, key_planes=None):
        super(Transformer, self).__init__()
        # 定义 AdaAttN 模块，用于执行自适应注意力和 AdaIN 操作,
        self.ada_attn_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.ada_attn_5_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes + 512)
        # 定义上采样和填充模块
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        # 定义卷积模块，用于将合并后的特征图转换为输出图像
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key,
                content5_1_key, style5_1_key, seed=None, content_masks=None, style_masks=None):
        return self.merge_conv(self.merge_conv_pad(
            self.ada_attn_4_1(
                content4_1, style4_1, content4_1_key, style4_1_key, seed, content_masks, style_masks) +
            self.upsample5_1(self.ada_attn_5_1(
                content5_1, style5_1, content5_1_key, style5_1_key, seed, content_masks, style_masks))))


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        # 定义 decoder_layer_1 层，包含一个反卷积层、一个 ReLU 激活函数和一个上采样层
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        # 定义 decoder_layer_2 层，包含多个反卷积层、ReLU 激活函数和填充层
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256 + 256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, cs, c_adain_3_feat):
        cs = self.decoder_layer_1(cs)
        cs = self.decoder_layer_2(torch.cat((cs, c_adain_3_feat), dim=1))
        return cs


def setup_args(parser):
    parser.add_argument(
        "--content_path", type=str, required=True,
        help="Path to a single content img",
    )
    parser.add_argument(
        "--style_path", type=str, required=True,
        help="Path to a single style img",
    )
    parser.add_argument(
        "--output_dir", type=str, default='output/',
        help="Output path",
    )
    parser.add_argument(
        "--resize", action='store_true',
        help="Whether resize images to the 512 scale, which is the training resolution "
             "of the model and may yield better performance"
    )
    parser.add_argument(
        "--keep_ratio", action='store_true',
        help="Whether keep the aspect ratio of original images while resizing"
    )


def main(args):
    """ Argument """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(args)

    """ Loading Input and Output Images """
    content_name = os.path.basename(args.content_path)
    style_name = os.path.basename(args.style_path)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, content_name[:content_name.rfind('.')] +
                               '_' + style_name[:style_name.rfind('.')] + '.jpg')
    content_im = cv2.imread(args.content_path)
    style_im = cv2.imread(args.style_path)
    original_h, original_w = content_im.shape[:2]
    if args.resize:
        content_im = resize(content_im, 512, args.keep_ratio)
        style_im = resize(style_im, 512, args.keep_ratio)
    h, w = content_im.shape[:2]
    h_s, w_s = style_im.shape[:2]

    """ Building Models """
    transformer_path = 'ckpt/latest_net_transformer.pth'
    decoder_path = 'ckpt/latest_net_decoder.pth'
    ada_attn_3_path = 'ckpt/latest_net_adaattn_3.pth'
    vgg_path = 'ckpt/vgg_normalised.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )
    # 将预训练的 VGG-19 模型的各个子层提取出来，以便在后续的图像风格迁移中使用。
    # 同时，也将这些子模型设置为评估模式，并将其参数的 requires_grad 属性设置为 False，以确保在训练时不会更新这些参数，从而提高模型的性能和训练速度。
    image_encoder.load_state_dict(torch.load(vgg_path))
    enc_layers = list(image_encoder.children())
    enc_1 = nn.Sequential(*enc_layers[:4]).to(device)
    enc_2 = nn.Sequential(*enc_layers[4:11]).to(device)
    enc_3 = nn.Sequential(*enc_layers[11:18]).to(device)
    enc_4 = nn.Sequential(*enc_layers[18:31]).to(device)
    enc_5 = nn.Sequential(*enc_layers[31:44]).to(device)
    image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
    for layer in image_encoder_layers:
        layer.eval()
        for p in layer.parameters():
            p.requires_grad = False
    transformer = Transformer(in_planes=512, key_planes=512 + 256 + 128 + 64).to(device)
    decoder = Decoder().to(device)
    ada_attn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 + 64, max_sample=64 * 64).to(device)
    transformer.load_state_dict(torch.load(transformer_path))
    decoder.load_state_dict(torch.load(decoder_path))
    ada_attn_3.load_state_dict(torch.load(ada_attn_3_path))
    transformer.eval()
    decoder.eval()
    ada_attn_3.eval()
    for p in transformer.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False
    for p in ada_attn_3.parameters():
        p.requires_grad = False

    def encode_with_intermediate(img):
        results = [img]
        for i in range(len(image_encoder_layers)):
            func = image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    def style_transfer():
        # 不对计算图进行梯度计算
        with torch.no_grad():
            # 使用 OpenCV 中的函数将风格图片转换为 PyTorch 张量，并将其发送到 GPU
            style = img_to_tensor(cv2.cvtColor(padding(style_im, 32), cv2.COLOR_BGR2RGB)).to(device)
            # 使用 OpenCV 中的函数将内容图片转换为 PyTorch 张量，并将其发送到 GPU
            content = img_to_tensor(cv2.cvtColor(padding(content_im, 32), cv2.COLOR_BGR2RGB)).to(device)
            # 使用 all_mask_c 变量生成内容图片的蒙版，并将其转换为 PyTorch 张量并发送到 GPU
            c_masks = [torch.from_numpy(padding(m, 32)).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                       for m in all_mask_c]
            # 使用 all_mask_s 变量生成风格图片的蒙版，并将其转换为 PyTorch 张量并发送到 GPU
            s_masks = [torch.from_numpy(padding(m, 32)).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)]
            # 对内容图片进行编码，并返回中间的特征图
            c_feats = encode_with_intermediate(content)
            # 对风格图片进行编码，并返回中间的特征图
            s_feats = encode_with_intermediate(style)
            # 对内容图片的第三个中间特征图和风格图片的第三个中间特征图进行 AdaIN 和注意力机制处理，并生成新的特征图
            c_adain_feat_3 = ada_attn_3(c_feats[2], s_feats[2], get_key(c_feats, 2), get_key(s_feats, 2), None,
                                        c_masks, s_masks)
            # 对内容图片和风格图片的中间特征图进行多尺度特征融合，并生成新的特征图
            cs = transformer(c_feats[3], s_feats[3], c_feats[4], s_feats[4], get_key(c_feats, 3), get_key(s_feats, 3),
                             get_key(c_feats, 4), get_key(s_feats, 4), None, c_masks, s_masks)
            # 使用风格图片的特征图作为参考，对新的特征图进行 AdaIN 处理，并进行上采样和卷积操作生成最终的风格化图像

            cs = decoder(cs, c_adain_feat_3)
            # 将张量转换回 numpy 数组，并将其转换为 RGB 格式
            cs = tensor_to_img(cs[:, :, :h, :w])
            cs = cv2.cvtColor(cs, cv2.COLOR_RGB2BGR)
            # 返回生成的风格化图像
            return cs

    """ Interaction """
    sam_checkpoint = "segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor_c = SamPredictor(sam)
    predictor_s = SamPredictor(sam)
    predictor_c.set_image(cv2.cvtColor(content_im, cv2.COLOR_BGR2RGB))
    predictor_s.set_image(cv2.cvtColor(style_im, cv2.COLOR_BGR2RGB))
    all_vis_c = [content_im.copy()]
    all_vis_s = [style_im.copy()]
    all_mask_c = []
    all_mask_s = []

    content_vis = content_im.copy()
    style_vis = style_im.copy()

    cv2.namedWindow('Content Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Content Image', content_vis)
    cv2.waitKey(1000)
    cv2.namedWindow('Style Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Style Image', style_vis)
    cv2.waitKey(1000)
    result = style_transfer()
    cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Result", result)
    cv2.waitKey(1000)

    def mouse_handle_sam(event, x, y, _1, _2):
        nonlocal predictor, vis, vis_, is_working, is_dragging, cur_points_labels, \
            cur_box, mask, window_name, color, is_valid

        if (not is_working) and (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN or (
                is_dragging and event == cv2.EVENT_MBUTTONUP)):
            is_working = True
            if event == cv2.EVENT_LBUTTONDOWN:
                cur_points_labels[0].append([x, y])
                cur_points_labels[1].append(1)
                cv2.circle(vis, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                cv2.imshow(window_name, vis)
            elif event == cv2.EVENT_RBUTTONDOWN:
                cur_points_labels[0].append([x, y])
                cur_points_labels[1].append(0)
                cv2.circle(vis, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.imshow(window_name, vis)
            elif event == cv2.EVENT_MBUTTONUP:
                cur_box += [x, y]
                if cur_box[0] > cur_box[2]:
                    cur_box[2] = cur_box[0]
                    cur_box[0] = x
                if cur_box[1] > cur_box[3]:
                    cur_box[3] = cur_box[1]
                    cur_box[1] = y
                is_dragging = False
                cv2.imshow(window_name, vis)
            mask, _, _ = predictor.predict(
                point_coords=np.array(cur_points_labels[0]) if len(cur_points_labels[0]) > 0 else None,
                point_labels=np.array(cur_points_labels[1]) if len(cur_points_labels[1]) > 0 else None,
                box=np.array(cur_box) if len(cur_box) == 4 else None,
                multimask_output=False
            )
            if not is_valid:
                is_valid = True

            mask = mask.reshape((mask.shape[1], mask.shape[2], 1))
            vis_ = (mask.astype(np.float) * (color * 0.6 + vis.astype(np.float) * 0.4) +
                    (1 - mask.astype(np.float)) * vis.astype(np.float)).astype(np.uint8)
            cv2.imshow(window_name, vis_)
            is_working = False
        elif not is_dragging and event == cv2.EVENT_MBUTTONDOWN:
            is_dragging = True
            cur_box = [x, y]
        elif is_dragging and event == cv2.EVENT_MOUSEMOVE:
            vis__ = vis_.copy()
            cv2.rectangle(vis__, (cur_box[0], cur_box[1]), (x, y), (255, 0, 0))
            cv2.imshow(window_name, vis__)

    def mouse_handle_circle(event, x, y, _1, _2):
        nonlocal is_dragging, point_a, x_max, x_min, y_max, y_min, vis_, window_name, mask, is_valid
        if is_dragging and event == cv2.EVENT_LBUTTONUP:
            is_dragging = False
            cv2.line(vis_, point_a, (x, y), (255, 255, 255))
            cv2.line(mask, point_a, (x, y), (255, 255, 255))
            point_a = (x, y)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
            cv2.imshow(window_name, vis_)
        elif is_dragging and event == cv2.EVENT_MOUSEMOVE:
            cv2.line(vis_, point_a, (x, y), (255, 255, 255))
            cv2.line(mask, point_a, (x, y), (255, 255, 255))
            point_a = (x, y)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
            cv2.imshow(window_name, vis_)
        elif not is_dragging and event == cv2.EVENT_LBUTTONDOWN:
            is_dragging = True
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
            point_a = (x, y)

    while True:
        print('Please Choose an Option for Content Image:')
        print('\t1: Select an Area by SAM')
        print('\t2: Specify an Area by a Contour')
        print('\t3: Undo Previous Content & Style Selection')
        print('\tOther: Finish!')
        option = input()
        if option == '1' or option == '2':
            is_working = False
            is_dragging = False
            is_valid = False
            cur_points_labels = ([], [])
            cur_box = []
            vis = content_vis
            vis_ = vis.copy()
            predictor = predictor_c
            color = (np.random.random(3) * 255).reshape(1, 1, 3)
            mask = np.zeros((h, w, 3)).astype(np.uint8)
            window_name = 'Content Image'

            if option == '1':
                print('\t\tLeft Clik on the Content Image to Set a Foreground Point;')
                print('\t\tRight Clik on the Content Image to Set a Background Point;')
                print('\t\tMiddle Clik on the Content Image and Drag Your Mouse to Specify a Bounding Box;')
                print('\t\tPress Any Key to Finish Your Current Selection')
                cv2.setMouseCallback('Content Image', mouse_handle_sam)
                cv2.waitKey(0)
            else:
                print('\t\tDrag Your Mouse to Draw a Contour;')
                print('\t\tPress Any Key to Finish Your Current Drawing')
                x_max = 0
                x_min = 1e10
                y_max = 0
                y_min = 1e10
                point_a = (0, 0)
                cv2.setMouseCallback('Content Image', mouse_handle_circle)
                cv2.waitKey(0)
                attempt = 0
                while True:
                    temp_mask = mask.copy()
                    seed_x = np.random.randint(x_min + 1, x_max)
                    seed_y = np.random.randint(y_min + 1, y_max)
                    cv2.floodFill(temp_mask, np.zeros((h + 2, w + 2)).astype(np.uint8), (seed_x, seed_y),
                                  (255, 255, 255), (50, 50, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
                    temp_mask = temp_mask.sum(axis=-1, keepdims=True).astype(np.bool)
                    if temp_mask.sum() <= h * w / 2:
                        is_valid = True
                        break
                    attempt += 1
                    if attempt > 500:
                        is_valid = False
                if is_valid:
                    mask = temp_mask
                    vis_ = (mask.astype(np.float) * color +
                            (1 - mask.astype(np.float)) * vis.astype(np.float)).astype(np.uint8)
                    cv2.imshow('Content Image', vis_)
                    cv2.waitKey(1000)
            if not is_valid:
                print('\t\tInvalid Selection! Please Re-try:')
            else:
                all_vis_c.append(vis_.copy())
                all_mask_c.append(mask)
                content_vis = vis_

                print('Please Choose an Option for Style Image:')
                while True:
                    print('\t1: Select an Area by SAM')
                    print('\t2: Specify an Area by a Contour')
                    print('\tOther: Undo Previous Content Selection')
                    option_s = input()

                    is_working = False
                    is_dragging = False
                    is_valid = False
                    cur_points_labels = ([], [])
                    cur_box = []
                    vis = style_vis
                    vis_ = vis.copy()
                    predictor = predictor_s
                    mask = np.zeros((h_s, w_s, 3)).astype(np.uint8)
                    window_name = 'Style Image'

                    if option_s == '1':
                        print('\t\tLeft Clik on the Style Image to Set a Foreground Point;')
                        print('\t\tRight Clik on the Style Image to Set a Background Point;')
                        print('\t\tMiddle Clik on the Style Image and Drag Your Mouse to Specify a Bounding Box;')
                        print('\t\tPress Any Key to Finish Your Current Selection')
                        cv2.setMouseCallback('Style Image', mouse_handle_sam, param='style')
                        cv2.waitKey(0)
                    elif option_s == '2':
                        print('\t\tDrag Your Mouse to Draw a Contour;')
                        print('\t\tPress Any Key to Finish Your Current Drawing')
                        x_max = 0
                        x_min = 1e10
                        y_max = 0
                        y_min = 1e10
                        point_a = (0, 0)
                        cv2.setMouseCallback('Style Image', mouse_handle_circle)
                        cv2.waitKey(0)
                        attempt = 0
                        while True:
                            temp_mask = mask.copy()
                            seed_x = np.random.randint(x_min + 1, x_max)
                            seed_y = np.random.randint(y_min + 1, y_max)
                            cv2.floodFill(temp_mask, np.zeros((h_s + 2, w_s + 2)).astype(np.uint8), (seed_x, seed_y),
                                          (255, 255, 255), (50, 50, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
                            temp_mask = temp_mask.sum(axis=-1, keepdims=True).astype(np.bool)
                            if temp_mask.sum() <= h_s * w_s / 2:
                                is_valid = True
                                break
                            attempt += 1
                            if attempt > 500:
                                is_valid = False
                        if is_valid:
                            mask = temp_mask
                            vis_ = (mask.astype(np.float) * color +
                                    (1 - mask.astype(np.float)) * vis.astype(np.float)).astype(np.uint8)
                            cv2.imshow('Style Image', vis_)
                            cv2.waitKey(1000)
                    else:
                        all_vis_c.pop()
                        all_mask_c.pop()
                        content_vis = all_vis_c[-1]
                        cv2.imshow("Content Image", content_vis)
                        cv2.waitKey(1000)
                        break
                    if is_valid:
                        all_vis_s.append(vis_.copy())
                        all_mask_s.append(mask)
                        style_vis = vis_
                        result = style_transfer()
                        cv2.imshow("Result", result)
                        cv2.waitKey(1000)
                        break
                    else:
                        print('\t\tInvalid Selection! Please Re-try:')
        elif option == '3':
            if len(all_vis_c) == 0 or len(all_mask_c) == 0:
                print('\t\tNo Previous Selection! Please Re-enter:')
            else:
                all_vis_c.pop()
                all_mask_c.pop()
                all_vis_s.pop()
                all_mask_s.pop()
                content_vis = all_vis_c[-1]
                style_vis = all_vis_s[-1]
                cv2.imshow("Content Image", content_vis)
                cv2.waitKey(1000)
                cv2.imshow("Style Image", style_vis)
                cv2.waitKey(1000)
                result = style_transfer()
                cv2.imshow("Result", result)
                cv2.waitKey(1000)
        else:
            break

    if args.resize:
        result = cv2.resize(result, (original_w, original_h))
    cv2.destroyAllWindows()
    cv2.imwrite(output_path, result)


if __name__ == '__main__':
    main(sys.argv[1:])
