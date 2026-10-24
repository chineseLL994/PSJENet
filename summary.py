# from KD.model_S import Student
# from KD.model_T import Teacher
#
# # file: profile_mccnet_smt.py
# import argparse
# import sys
# import torch
# import torch.nn as nn
#
#
# def count_params(model: nn.Module):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# def try_profile_with_thop(model, inputs):
#     try:
#         from thop import profile, clever_format
#     except Exception as e:
#         return None, None, f"THOP 不可用：{e}"
#     try:
#         flops, params = profile(model, inputs=(inputs,), verbose=False)
#         macs = flops / 2  # 部分论文按 MACs 计，这里给个参考（可忽略）
#         flops_str, params_str = clever_format([flops, params], "%.3f")
#         return (flops, params, macs), (flops_str, params_str), None
#     except Exception as e:
#         return None, None, f"THOP profiling 失败：{e}"
#
# def try_profile_with_fvcore(model, inputs):
#     try:
#         from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
#     except Exception as e:
#         return None, None, f"fvcore 不可用：{e}"
#     try:
#         model.eval()
#         with torch.no_grad():
#             flops_ana = FlopCountAnalysis(model, inputs)
#             flops = flops_ana.total()
#             params = sum(p.numel() for p in model.parameters())
#             table = flop_count_table(flops_ana, max_depth=3)
#         return (flops, params, None), table, None
#     except Exception as e:
#         return None, None, f"fvcore profiling 失败：{e}"
#
# def main():
#     parser = argparse.ArgumentParser(description="Profile MCCNet_SMT FLOPs & Params")
#     parser.add_argument("--height", type=int, default=384, help="输入高度（默认352，按你的项目修改）")
#     parser.add_argument("--width", type=int, default=384, help="输入宽度（默认352，按你的项目修改）")
#     parser.add_argument("--channels", type=int, default=3, help="输入通道数（默认3）")
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--use-fp16", action="store_true", help="在支持的设备上用半精度推理（可稍微加速/省显存）")
#     args = parser.parse_args()
#
#     device = torch.device(args.device)
#     # model = MCCNet_Student()  # 如需自定义 init 参数，在这里改
#     model =Student()
#     model.eval().to(device)
#
#     dummy = torch.randn(1, args.channels, args.height, args.width, device=device)
#     if args.use_fp16 and device.type == "cuda":
#         model = model.half()
#         dummy = dummy.half()
#
#     # 1) 统计参数量（无依赖）
#     total_params = count_params(model)
#     print("=" * 60)
#     print(f"Trainable Params: {total_params:,}  ({total_params/1e6:.3f} M)")
#
#     # 2) 优先用 THOP
#     print("-" * 60)
#     print("尝试使用 THOP 统计 FLOPs ...")
#     thop_vals, thop_fmt, thop_err = try_profile_with_thop(model, dummy)
#     if thop_err is None:
#         flops, params, macs = thop_vals
#         flops_g = flops / 1e9
#         macs_g = macs / 1e9 if macs is not None else None
#         print(f"THOP FLOPs: {flops:,}  ({flops_g:.3f} GFLOPs)")
#         if macs_g is not None:
#             print(f"THOP MACs:  {macs:,}  ({macs_g:.3f} GMACs)  [注：仅供参考]")
#         print(f"THOP Params: {params:,}  ({params/1e6:.3f} M)")
#         print("-" * 60)
#         print("提示：不同工具对某些算子的 FLOPs 统计方法略有差异，属正常现象。")
#         return
#     else:
#         print(thop_err)
#
#     # 3) 回退到 fvcore
#     print("-" * 60)
#     print("尝试使用 fvcore 统计 FLOPs ...")
#     fv_vals, fv_table, fv_err = try_profile_with_fvcore(model, dummy)
#     if fv_err is None:
#         flops, params, _ = fv_vals
#         print(f"fvcore FLOPs: {flops:,}  ({flops/1e9:.3f} GFLOPs)")
#         print(f"fvcore Params: {params:,}  ({params/1e6:.3f} M)")
#         print("-" * 60)
#         print("fvcore 逐层统计表（截断深度=3，仅供参考）：")
#         print(fv_table)
#         print("-" * 60)
#         print("提示：不同工具对某些算子的 FLOPs 统计方法略有差异，属正常现象。")
#         return
#     else:
#         print(fv_err)
#         print("=" * 60)
#         print("THOP 与 fvcore 均不可用/失败，请检查依赖并确保模型 forward 能在给定输入尺寸下正常运行。")
#         print("建议：pip install thop fvcore && 再运行本脚本。")
#         sys.exit(1)
#
# if __name__ == "__main__":
#     main()
#



import argparse
import sys
import torch
import torch.nn as nn
from sam2_model.rgbT.SAM_AC3U_score import Model as MyNet

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--img_size', type=int, default=512, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
parser.add_argument('--freeze_image_encoder', type=bool, default=True, help='True or False')
parser.add_argument('--save_path', type=str, default='./ViDSOD-100_ckp/RGBDV/MyNet_AC3U_GBD/',
                    help='the path to save models and logs')
parser.add_argument('--sam2_cfg', type=str, default='D:/Chinese_DS/SAMSOD-main/sam2_configs/sam2_hiera_l.yaml',
                    help='train from checkpoints')
parser.add_argument('--sam2_checkpoint', type=str, default='D:/Chinese_DS/SAMSOD-main/sam2_hiera_large.pt',
                    help='train from checkpoints')
parser.add_argument('--score_pkl_path', type=str,
                    default='D:/Chinese_DS/SAMSOD-main/Datasets/ViDSOD-100/train/vidsod100_depth_score.pkl',
                    help='path to quality score pkl file')
parser.add_argument('--gate_loss_weight', type=float, default=0.1,
                    help='weight for gate quality prediction loss')
parser.add_argument('--frame_length', type=int, default=10)
opt = parser.parse_args()

def count_params(model: nn.Module):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def try_profile_with_thop(model, inputs):
    try:
        from thop import profile, clever_format
    except Exception as e:
        return None, None, f"THOP 不可用：{e}"

    try:
        # 确保我们将输入传递为元组
        flops, params = profile(model, inputs=inputs, verbose=False)
        macs = flops / 2  # 部分论文按 MACs 计，这里给个参考（可忽略）
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return (flops, params, macs), (flops_str, params_str), None
    except Exception as e:
        return None, None, f"THOP profiling 失败：{e}"


def main():
    parser = argparse.ArgumentParser(description="Profile MCCNet_SMT FLOPs & Params")
    parser.add_argument("--height", type=int, default=512, help="输入高度（默认384，按你的项目修改）")
    parser.add_argument("--width", type=int, default=512, help="输入宽度（默认384，按你的项目修改）")
    parser.add_argument("--channels", type=int, default=3, help="输入通道数（默认3）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-fp16", action="store_true", help="在支持的设备上用半精度推理（可稍微加速/省显存）")
    args = parser.parse_args()

    device = torch.device(args.device)
    # model = Student()  # 学生模型
    # model = Teacher()  # 教师模型
    model = MyNet(opt  )
    model.eval().to(device)

    # 创建两个输入的虚拟张量：一个 RGB 图像 (3, H, W)，一个深度图像 (3, H, W)
    dummy_rgb = torch.randn(1, 3, args.height, args.width, device=device)  # RGB 输入
    dummy_depth = torch.randn(1, 3, args.height, args.width, device=device)  # 深度图输入
    # dummy_f = torch.randn(1, 3, args.height, args.width, device=device)  # 深度图输入

    if args.use_fp16 and device.type == "cuda":
        model = model.half()
        dummy_rgb = dummy_rgb.half()
        dummy_depth = dummy_depth.half()
        # dummy_f = dummy_f.half()

    # 1) 统计参数量（无依赖）
    total_params = count_params(model)
    print("=" * 60)
    print(f"Trainable Params: {total_params:,}  ({total_params / 1e6:.3f} M)")

    # 2) 优先用 THOP
    print("-" * 60)
    print("尝试使用 THOP 统计 FLOPs ...")
    thop_vals, thop_fmt, thop_err = try_profile_with_thop(model, (dummy_rgb, dummy_depth))  # 确保传递元组
    if thop_err is None:
        flops, params, macs = thop_vals
        flops_g = flops / 1e9
        macs_g = macs / 1e9 if macs is not None else None
        print(f"THOP FLOPs: {flops:,}  ({flops_g:.3f} GFLOPs)")
        if macs_g is not None:
            print(f"THOP MACs:  {macs:,}  ({macs_g:.3f} GMACs)  [注：仅供参考]")
        print(f"THOP Params: {params:,}  ({params / 1e6:.3f} M)")
        print("-" * 60)
        print("提示：不同工具对某些算子的 FLOPs 统计方法略有差异，属正常现象。")
        return
    else:
        print(thop_err)

    # 3) 回退到 fvcore
    print("-" * 60)
    print("尝试使用 fvcore 统计 FLOPs ...")
    fv_vals, fv_table, fv_err = try_profile_with_fvcore(model, (dummy_rgb, dummy_depth))
    if fv_err is None:
        flops, params, _ = fv_vals
        print(f"fvcore FLOPs: {flops:,}  ({flops / 1e9:.3f} GFLOPs)")
        print(f"fvcore Params: {params:,}  ({params / 1e6:.3f} M)")
        print("-" * 60)
        print("fvcore 逐层统计表（截断深度=3，仅供参考）：")
        print(fv_table)
        print("-" * 60)
        print("提示：不同工具对某些算子的 FLOPs 统计方法略有差异，属正常现象。")
        return
    else:
        print(fv_err)
        print("=" * 60)
        print("THOP 与 fvcore 均不可用/失败，请检查依赖并确保模型 forward 能在给定输入尺寸下正常运行。")
        print("建议：pip install thop fvcore && 再运行本脚本。")
        sys.exit(1)


if __name__ == "__main__":
    main()
