# ========== 修改后的推理代码 ==========
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
# from sam2_model.rgbT.SAM_ACI_score import Model
# from sam2_model.Ablation.woACI import Model
# from sam2_model.Ablation.woGBDecoder import Model
# from sam2_model.Ablation.woQA import Model
# from sam2_model.Ablation.woACI_RGBguide import Model
# from sam2_model.Ablation.woACI_Depthguide import Model
# from sam2_model.Ablation.woACI_asymmetric import Model
# from sam2_model.Ablation.MAGNet_decoder import Model
# from sam2_model.Ablation.woDC import Model
from sam2_model.rgbT.SAM_AC3U_score import Model
# from sam2_model.Ablation.RGBD_Cat import Model
# from sam2_model.Ablation.acqc_full_ablation.model_acqc_ablation import Model




from Datasets.get_data_rgbdvsod import get_test_loader   # 注意：这里导入的是您重新定义的 get_test_loader
import argparse

# 设置设备
torch.cuda.set_device(0)

# 解析参数（保持原有）
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=45, help='epoch number')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--img_size', type=int, default=512, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=140, help='every n epochs decay learning rate')
parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
parser.add_argument('--freeze_image_encoder', type=bool, default=True, help='True or False')
parser.add_argument('--save_path', type=str, default='./log/', help='the path to save models and logs')
parser.add_argument('--sam2_cfg', type=str, default='D:/Chinese_DS/SAMSOD-main/sam2_configs/sam2_hiera_l.yaml',
                    help='train from checkpoints')
parser.add_argument('--sam2_checkpoint', type=str, default='D:/Chinese_DS/SAMSOD-main/sam2_hiera_large.pt',
                    help='train from checkpoints')

opt = parser.parse_args()

# 初始化模型并加载权重
model = Model(opt)
# model.load_state_dict(torch.load('/home/sutai/LYW/RDVS_ckp/Ablation(seed101)/sam_rgbt0.022407426best_8.pth'))
model.load_state_dict(torch.load('D:/Chinese_DS/SAMSOD-main/DVISAL_ckp/Ablation/Revise/GussDepth_40(woCMDQA)/sam_rgbt0.03935106051312324_1.pth'))

model.cuda()
model.eval()


# 测试数据集
# 测试数据集
test_datasets = ['DVisal']
# data_root = 'D:/Chinese_DS/SAMSOD-main/Datasets/RDVS/RDVS-TEST'
data_root = 'D:/Chinese_DS/SAMSOD-main/Datasets/dataset/DVisal/test'
output_base = 'SOD_Evaluation_Metrics_DVisal/PreOut/MyNet(GussDepth40_woCMDQA)'

for dataset in test_datasets:
    # 最终输出根目录: .../PreOut/SAMSOD/vidsod_100/
    save_root = os.path.join(output_base, dataset)
    os.makedirs(save_root, exist_ok=True)

    # 创建测试 DataLoader（注意修改后的 test_dataset 需要接收 root 参数）
    test_loader = get_test_loader(
        root=data_root,
        batchsize=1,
        testsize=opt.img_size
    )

    for batch_idx, (image, gt, depth, rel_name) in enumerate(test_loader):
        # rel_name 是元组，例如 ('subfolder1/frame_001.png',)
        rel_path = rel_name[0]  # 字符串：subfolder1/frame_001.png
        # 分离子文件夹和文件名
        subfolder, filename = os.path.split(rel_path)

        # 创建目标子文件夹
        target_dir = os.path.join(save_root, subfolder)
        os.makedirs(target_dir, exist_ok=True)

        # 完整保存路径
        full_save_path = os.path.join(target_dir, filename)

        # 推理
        image = image.cuda()
        depth = depth.cuda()
        with torch.no_grad():
            res, _ = model(image, depth)

        # 后处理
        res = F.interpolate(res, size=gt.shape[2:], mode='bilinear', align_corners=False)
        res = torch.sigmoid(res).squeeze().cpu().numpy()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        save_img = (res * 255).astype(np.uint8)

        # 保存
        cv2.imwrite(full_save_path, save_img)
        print(f"已保存: {full_save_path}")

print("所有测试集处理完成！")

