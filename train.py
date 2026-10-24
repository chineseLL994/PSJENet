# import argparse
# import logging
# import os
# from datetime import datetime
# import cv2
# from Datasets.get_data_rgbdvsod3 import get_loader, get_test_loader
# # from Datasets.data_score import get_loader, get_test_loader
# from loss import smoothness
# from loss.lscloss import *
# # from sam2_model.ACImodels.ACINet import ACINet as Model
# from sam2_model.rgbT.SAM_ACI_score import Model
# from tools.tools import *
# from optim.gmd_name import GMD
#
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--epoch', type=int, default=200, help='epoch number')
# parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
# parser.add_argument('--img_size', type=int, default=512, help='training dataset size')
# parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
# parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
# parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
# parser.add_argument('--freeze_image_encoder', type=bool, default=True, help='True or False')
# parser.add_argument('--save_path', type=str, default='./RDVS_ckp/Ablation/MyNet2/',
#                     help='the path to save models and logs')
# parser.add_argument('--sam2_cfg', type=str, default='D:/Chinese_DS/SAMSOD-main/sam2_configs/sam2_hiera_l.yaml',
#                     help='train from checkpoints')
# parser.add_argument('--sam2_checkpoint', type=str, default='D:/Chinese_DS/SAMSOD-main/sam2_hiera_large.pt',
#                     help='train from checkpoints')
# parser.add_argument('--score_pkl_path', type=str,
#                     default='D:/Chinese_DS/SAMSOD-main/Datasets/RDVS/RDVS-TRAIN/RDVS_depth_score.pkl',
#                     help='path to quality score pkl file')
# parser.add_argument('--gate_loss_weight', type=float, default=0.1,
#                     help='weight for gate quality prediction loss')
# parser.add_argument('--frame_length', type=int, default=10)
# opt = parser.parse_args()
#
# # 定义日志文件路径
# save_path = opt.save_path
# log_filename = 'log.txt'
# full_log_path = os.path.join(save_path, log_filename)
#
# # 确保日志目录存在
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
#
# # 配置日志
# logging.basicConfig(filename=full_log_path, level=logging.INFO,
#                     format='%(asctime)s - %(message)s')
#
# print('Learning Rate: {}'.format(opt.lr))
# model = Model(opt)
# model.setup()
# # model.load_pre(opt.load)
# model.cuda()
# params_with_names = []
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         # print(name)
#         params_with_names.append({'name': name, 'param': param})
#
# optimizer = torch.optim.Adam([p['param'] for p in params_with_names], lr=0.001)
#
# for group, param_with_name in zip(optimizer.param_groups, params_with_names):
#     group['name'] = [p['name'] for p in params_with_names]
#
# gmd = GMD(optimizer, reduction='mean')
#
# # 训练数据加载器（添加 score_pkl_path 参数）
# train_loader = get_loader(root='D:/Chinese_DS/SAMSOD-main/Datasets/RDVS/RDVS-TRAIN',
#                           batchsize=opt.batchsize,
#                           trainsize=opt.img_size,
#                           num_workers=4,
#                           score_pkl_path=opt.score_pkl_path)
#
# # 测试数据加载器
# test_loader = get_test_loader(root='D:/Chinese_DS/SAMSOD-main/Datasets/RDVS/RDVS-TEST',
#                               batchsize=1,
#                               testsize=opt.img_size)
# # train_loader = get_loader(root='D:/Chinese_DS/SAMSOD-main/Datasets/vidsod_100/train',
# #                           batchsize=opt.batchsize,
# #                           trainsize=opt.img_size,
# #                           num_workers=4,
# #                           score_pkl_path=opt.score_pkl_path)
# #
# # # 测试数据加载器
# # test_loader = get_test_loader(root='D:/Chinese_DS/SAMSOD-main/Datasets/vidsod_100/test',
# #                               batchsize=1,
# #                               testsize=opt.img_size)
#
# total_step = len(train_loader)
#
# # 损失函数
# CE = torch.nn.BCELoss()
# smooth_loss = smoothness.smoothness_loss(size_average=True)
# best_mae = 1
# best_epoch = 0
#
# loss_lsc = LocalSaliencyCoherence().cuda()
# loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
# loss_lsc_radius = 5
# save_path = opt.save_path
#
# def structure_loss(pred, mask):
#     weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
#     wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
#
#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask) * weit).sum(dim=(2, 3))
#     union = ((pred + mask) * weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1) / (union - inter + 1)
#     return (wbce + wiou).mean()
#
#
# def train(train_loader, model, gmd, epoch):
#     model.train()
#
#     for i, pack in enumerate(train_loader, start=1):
#         gmd.zero_grad()
#
#         # 解包数据（根据新的返回值）
#         images_rgb, aug_image, images_td, aug_modal,  gts, depth_gate_gt = pack
#
#         aug_image = aug_image.cuda()
#         aug_modal = aug_modal.cuda()
#         # aug_flow = aug_flow.cuda()
#         gts = gts.cuda()
#         depth_gate_gt = depth_gate_gt.cuda()
#         # flow_gate_gt = flow_gate_gt.cuda()
#
#         # 前向传播
#         SAM_mask1,  depth_gate = model(aug_image, aug_modal)
#         # SAM_mask1, SAM_mask2 = model(aug_image, aug_modal)
#
#         # 计算分割损失
#         loss_seg = structure_loss(SAM_mask1, gts)
#         # loss_seg2 = structure_loss(SAM_mask2, gts)
#         # loss_seg3 = structure_loss(SAM_mask3, gts)
#         d_gate_loss = F.smooth_l1_loss(depth_gate, depth_gate_gt)
#
#         loss_total = loss_seg + 0.5 * d_gate_loss
#         # loss_total = loss_seg + loss_seg2
#
#         gmd.pc_backward([loss_total], model)
#         gmd.step()
#
#         if i % 100 == 0 or i == total_step or i == 1:
#             now_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_seg: {:.4f}'.
#                   format(now_formatted, epoch, opt.epoch, i, total_step, loss_seg.data))
#             logging.info(
#                 '#TRAIN#: Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_seg: {:.4f}'.
#                 format(epoch, opt.epoch, i, total_step, loss_seg.data))
#
#         del loss_seg, loss_total, SAM_mask1
#
#         torch.cuda.empty_cache()
#
#
# def test(test_loader, model, epoch, save_path):
#     global best_mae, best_epoch
#     model.eval()
#     mae_sum = 0
#     total_samples = len(test_loader)
#
#     with torch.no_grad():
#         for pack in test_loader:
#             # 解包测试数据（根据新的返回值）
#             image, gt, depth, name = pack
#
#             image = image.cuda()
#             depth = depth.cuda()
#             # flow = flow.cuda()
#
#             # 模型前向
#             pred, _ = model(image, depth)
#
#             # 转换 GT 为 numpy
#             gt_np = gt.squeeze().cpu().numpy()
#             if gt_np.ndim == 3:
#                 gt_np = gt_np.squeeze(0)
#
#             # 转换预测
#             pred_np = pred.sigmoid().detach().cpu().numpy().squeeze()
#             if pred_np.shape != gt_np.shape:
#                 pred_np = cv2.resize(pred_np, (gt_np.shape[1], gt_np.shape[0]))
#             # 归一化
#             pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
#
#             mae = np.sum(np.abs(pred_np - gt_np)) / (gt_np.shape[0] * gt_np.shape[1])
#             mae_sum += mae
#
#     mae = mae_sum / total_samples
#     print('Epoch: {} MAE: {:.10f} ####  bestMAE: {:.10f} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
#
#     # 保存模型逻辑不变
#     if epoch == 1:
#         best_mae = mae
#         best_epoch = epoch
#         torch.save(model.state_dict(), save_path + 'sam_rgbt' + str(mae) + '_%d' % epoch + '.pth')
#     else:
#         if mae < best_mae:
#             best_mae = mae
#             best_epoch = epoch
#             torch.save(model.state_dict(),
#                        save_path + 'sam_rgbt' + str(best_mae) + 'best' + '_%d' % epoch + '.pth')
#             print('best epoch:{}'.format(epoch))
#         if mae < 0.025:
#             torch.save(model.state_dict(),
#                        save_path + 'sam_rgbt' + str(mae) + 'best' + '_%d' % epoch + '.pth')
#     logging.info('#TEST#:Epoch:{} MAE:{:.10f} bestEpoch:{} bestMAE:{:.10f}'.
#                  format(epoch, mae, best_epoch, best_mae))
#
#
# if __name__ == '__main__':
#     print("Start Training!")
#     logging.info("训练rgbt")
#
#     for epoch in range(1, opt.epoch + 1):
#         train(train_loader, model, gmd, epoch)
#         test(test_loader, model, epoch, save_path)

import argparse
import logging
import os
from datetime import datetime
import cv2
from Datasets.get_data_rgbdvsod3 import get_loader, get_test_loader
# from Datasets.data_score import get_loader, get_test_loader
from loss import smoothness
from loss.lscloss import *
# from sam2_model.Ablation.acqc_full_ablation.model_acqc_ablation import Model
from sam2_model.rgbT.SAM_AC3U_score import Model
from tools.tools import *
from optim.gmd_name import GMD

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--img_size', type=int, default=512, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
parser.add_argument('--freeze_image_encoder', type=bool, default=True, help='True or False')
# parser.add_argument('--save_path', type=str, default='./DVisal_ckp/ABLation/Revise/seed1/',
#                     help='the path to save models and logs')
# parser.add_argument('--sam2_cfg', type=str, default='D:/Chinese_DS/SAMSOD-main/sam2_configs/sam2_hiera_l.yaml',
#                     help='train from checkpoints')
# parser.add_argument('--sam2_checkpoint', type=str, default='D:/Chinese_DS/SAMSOD-main/sam2_hiera_large.pt',
#                     help='train from checkpoints')
# parser.add_argument('--score_pkl_path', type=str,
#                     default='D:/Chinese_DS/SAMSOD-main/Datasets/dataset/DVisal/train/DVisal_depth_score.pkl',
#                     help='path to quality score pkl file')
parser.add_argument('--save_path', type=str, default='./DVISAL_ckp/ABLation/Revise/GussDepth_40/',
                    help='the path to save models and logs')
parser.add_argument('--sam2_cfg', type=str, default='D:/Chinese_DS/SAMSOD-main/sam2_configs/sam2_hiera_l.yaml',
                    help='train from checkpoints')
parser.add_argument('--sam2_checkpoint', type=str, default='D:/Chinese_DS/SAMSOD-main/sam2_hiera_large.pt',
                    help='train from checkpoints')
parser.add_argument('--score_pkl_path', type=str,
                    default='D:\\Chinese_DS\\SAMSOD-main\\Datasets\\dataset\\DVisal\\train\\Dvisal_Gaussian40_depth_score.pkl',
                    help='path to quality score pkl file')
parser.add_argument('--gate_loss_weight', type=float, default=0.1,
                    help='weight for gate quality prediction loss')
parser.add_argument('--frame_length', type=int, default=10)
parser.add_argument(
    '--acqc_variant',
    type=str,
    default='Q4',
    choices=[
        'G0', 'G1', 'G2', 'G3', 'G4',
        'Q0', 'Q1', 'Q2', 'Q3', 'Q4',
        'M0', 'M1', 'M2', 'M3', 'M4'
    ],
    help='ACQC quality gating ablation variant'
)
opt = parser.parse_args()

# 定义日志文件路径
save_path = opt.save_path
log_filename = 'log.txt'
full_log_path = os.path.join(save_path, log_filename)

# 确保日志目录存在
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 配置日志
logging.basicConfig(filename=full_log_path, level=logging.INFO,
                    format='%(asctime)s - %(message)s')

print('Learning Rate: {}'.format(opt.lr))
model = Model(opt)
model.setup()
# model.load_pre(opt.load)
model.cuda()
params_with_names = []
for name, param in model.named_parameters():
    if param.requires_grad:
        # print(name)
        params_with_names.append({'name': name, 'param': param})

optimizer = torch.optim.Adam([p['param'] for p in params_with_names], lr=0.001)

for group, param_with_name in zip(optimizer.param_groups, params_with_names):
    group['name'] = [p['name'] for p in params_with_names]

gmd = GMD(optimizer, reduction='mean')

# 训练数据加载器（添加 score_pkl_path 参数）
# RGB 和 GT 数据集路径
train_root = r'D:/Chinese_DS/SAMSOD-main/Datasets/dataset/DVisal/train'


# 训练数据加载器
train_loader = get_loader(
    root=train_root,
    batchsize=opt.batchsize,
    trainsize=opt.img_size,
    num_workers=4,
    score_pkl_path=opt.score_pkl_path
)

# 测试数据加载器
test_loader = get_test_loader(root='D:/Chinese_DS/SAMSOD-main/Datasets/dataset/DVisal/test/',
                              batchsize=1,
                              testsize=opt.img_size)

total_step = len(train_loader)

# 损失函数
CE = torch.nn.BCELoss()
smooth_loss = smoothness.smoothness_loss(size_average=True)
best_mae = 1
best_epoch = 0

loss_lsc = LocalSaliencyCoherence().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
save_path = opt.save_path

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, gmd, epoch):
    model.train()

    for i, pack in enumerate(train_loader, start=1):
        gmd.zero_grad()

        # ---------- NEW: 解包增加光流数据 ----------
        # 数据集返回顺序: image, aug_image, depth, aug_depth, flow, aug_flow, gt, depth_gate_gt
        images_rgb, aug_image, images_td, aug_modal, flow_orig, aug_flow, gts, depth_gate_gt = pack

        aug_image = aug_image.cuda()
        aug_modal = aug_modal.cuda()
        # aug_flow = aug_flow.cuda()
        gts = gts.cuda()
        depth_gate_gt = depth_gate_gt.cuda()

        SAM_mask1, depth_gate = model(aug_image, aug_modal)
        # SAM_mask1 = model(aug_image, aug_modal)

        # 计算分割损失
        loss_seg = structure_loss(SAM_mask1, gts)
        # 深度质量分数回归损失
        d_gate_loss = F.smooth_l1_loss(depth_gate, depth_gate_gt)

        # 实验 1
        # d_gate_loss = F.l1_loss(depth_gate, depth_gate_gt)

        # 实验 2
        # d_gate_loss = F.mse_loss(depth_gate, depth_gate_gt)

        # 实验 3
        # d_gate_loss = F.huber_loss(depth_gate, depth_gate_gt, delta=0.5)

        # 总损失 (光流gate暂未使用)
        loss_total = loss_seg + 0.5 * d_gate_loss
        # loss_total = loss_seg

        gmd.pc_backward([loss_total], model)
        gmd.step()

        if i % 100 == 0 or i == total_step or i == 1:
            now_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_seg: {:.4f}'.
                  format(now_formatted, epoch, opt.epoch, i, total_step, loss_seg.data))
            logging.info(
                '#TRAIN#: Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_seg: {:.4f}'.
                format(epoch, opt.epoch, i, total_step, loss_seg.data))

        del loss_seg, loss_total, SAM_mask1

        torch.cuda.empty_cache()


def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    mae_sum = 0
    total_samples = len(test_loader)

    with torch.no_grad():
        for pack in test_loader:
            # ---------- NEW: 解包测试数据增加光流 ----------
            # 测试集返回顺序: image, gt, depth, flow, name
            image, gt, depth, flow, name = pack

            image = image.cuda()
            depth = depth.cuda()
            # flow = flow.cuda()          # NEW: 光流图移到GPU

            # 模型前向 (增加光流)
            pred, _ = model(image, depth)
            # pred = model(image, depth)

            # 转换 GT 为 numpy
            gt_np = gt.squeeze().cpu().numpy()
            if gt_np.ndim == 3:
                gt_np = gt_np.squeeze(0)

            # 转换预测
            pred_np = pred.sigmoid().detach().cpu().numpy().squeeze()
            if pred_np.shape != gt_np.shape:
                pred_np = cv2.resize(pred_np, (gt_np.shape[1], gt_np.shape[0]))
            # 归一化
            pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)

            mae = np.sum(np.abs(pred_np - gt_np)) / (gt_np.shape[0] * gt_np.shape[1])
            mae_sum += mae

    mae = mae_sum / total_samples
    print('Epoch: {} MAE: {:.10f} ####  bestMAE: {:.10f} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))

    # 保存模型逻辑不变
    if epoch == 1:
        best_mae = mae
        best_epoch = epoch
        torch.save(model.state_dict(), save_path + 'sam_rgbt' + str(mae) + '_%d' % epoch + '.pth')
    else:
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            torch.save(model.state_dict(),
                       save_path + 'sam_rgbt' + str(best_mae) + 'best' + '_%d' % epoch + '.pth')
            print('best epoch:{}'.format(epoch))
        if mae < 0.025:
            torch.save(model.state_dict(),
                       save_path + 'sam_rgbt' + str(mae) + 'best' + '_%d' % epoch + '.pth')
    logging.info('#TEST#:Epoch:{} MAE:{:.10f} bestEpoch:{} bestMAE:{:.10f}'.
                 format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start Training!")
    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, gmd, epoch)
        test(test_loader, model, epoch, save_path)