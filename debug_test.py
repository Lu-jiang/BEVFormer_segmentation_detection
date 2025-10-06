# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import sys
sys.path.insert(0, '/home/jianglu/Documents/BEVFormer_segmentation_detection')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

#from mmdet3d.apis import single_gpu_test
from projects.mmdet3d_plugin.apis.test import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.apis.test import multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', default='projects/configs/bevformer/bevformer_base_seg_det_150x150.py', help='test config file path')
    parser.add_argument('--checkpoint', default='ckpts/bevformer_base_seg_det_150.pth', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        #nargs='+',
        default=True,
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', default='/home/jianglu/Documents/BEVFormer_segmentation_detection/visual_work_dir', help='directory where results will be saved')
    #parser.add_argument('--show-dir', default=None, help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    # 调用 parse_args 函数解析命令行参数
    args = parse_args()

    # 检查是否至少指定了一个操作（保存结果、评估、格式化结果、显示结果等）
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')
    # 原理：确保用户在运行脚本时指定了至少一个有意义的操作，避免无目的的运行。
    # 目的：防止用户忘记指定操作而导致脚本执行无结果。

    # 检查 --eval 和 --format_only 参数是否同时指定
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    # 原理：这两个参数的功能互斥，同时指定会造成逻辑冲突。
    # 目的：避免用户在命令行中同时输入这两个冲突的参数，保证脚本逻辑的正确性。

    # 检查 --out 参数指定的输出文件是否为 .pkl 或 .pickle 文件
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    # 原理：代码中后续使用的保存结果的方式通常是将数据以 pickle 格式保存。
    # 目的：确保用户指定的输出文件格式正确，以便后续正确保存结果。

    # 从指定的配置文件中加载配置
    cfg = Config.fromfile(args.config)
    # 原理：使用 mmcv 的 Config 类从文件中读取配置信息，方便后续使用。
    # 目的：获取模型、数据集等的配置参数。

    # 如果命令行中指定了配置选项，则将其合并到配置中
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # 原理：允许用户在命令行中临时修改配置文件中的某些参数。
    # 目的：提供灵活性，方便用户在不修改配置文件的情况下调整参数。

    # 如果配置文件中指定了自定义导入模块，则导入这些模块
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # 原理：支持用户自定义导入模块，扩展功能。
    # 目的：可以引入用户自定义的数据集、模型等模块。

    # 如果配置文件中指定了插件，则导入插件模块
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
    # 原理：允许用户通过插件的方式扩展功能。
    # 目的：可以引入自定义的数据集构建、模型评估等功能。

    # 根据配置文件中的设置，开启或关闭 cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # 原理：cudnn_benchmark 可以自动寻找最适合当前配置的高效算法，提高计算效率。
    # 目的：在输入尺寸固定的情况下，加速模型的训练和推理。

    # 将模型的预训练参数设置为 None，因为在测试阶段不需要预训练参数
    cfg.model.pretrained = None
    # 原理：避免在测试阶段加载不必要的预训练参数。
    # 目的：确保测试时使用的是指定的检查点文件中的参数。

    # 处理测试数据集配置
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    # 原理：将测试数据集设置为测试模式，处理每个GPU的样本数，并根据样本数调整数据处理流程。
    # 目的：确保测试数据集的配置正确，适应不同的测试场景。

    # 根据 --launcher 参数决定是否使用分布式训练
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # 原理：根据用户指定的启动器决定是否使用分布式训练，并初始化分布式环境。
    # 目的：支持单GPU和多GPU测试。

    # 如果指定了随机种子，则设置随机种子以确保结果的可重复性
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    # 原理：通过设置随机种子，使得每次运行代码时的随机操作结果相同。
    # 目的：方便结果的复现和比较。

    # 构建测试数据集
    dataset = build_dataset(cfg.data.test)
    # 原理：根据配置文件中的数据集配置构建测试数据集对象。
    # 目的：准备好用于测试的数据集。

    # 构建数据加载器
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    # 原理：根据数据集、样本数、工作进程数等参数构建数据加载器，用于批量加载数据。
    # 目的：高效地将测试数据加载到模型中进行推理。

    # 将模型的训练配置设置为 None，因为在测试阶段不需要训练配置
    cfg.model.train_cfg = None
    # 原理：避免在测试阶段使用不必要的训练配置。
    # 目的：确保测试时只使用测试配置。

    # 根据配置文件构建模型
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 原理：根据配置文件中的模型配置和测试配置构建模型对象。
    # 目的：创建用于测试的模型。

    # 如果配置文件中指定了 fp16 训练，则将模型转换为 fp16 模式
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # 原理：使用半精度浮点数（fp16）可以减少内存占用和计算量，提高推理速度。
    # 目的：在支持的硬件上加速模型推理。

    # 加载预训练的检查点文件
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # 原理：从指定的检查点文件中加载模型的参数。
    # 目的：使用训练好的模型进行测试。

    # 如果指定了融合卷积和 BN 层，则进行融合操作
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # 原理：融合卷积和 BN 层可以减少计算量，提高推理速度。
    # 目的：进一步优化模型的推理性能。

    # 处理类别信息
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # 原理：将类别信息添加到模型中，方便后续的结果处理和可视化。
    # 目的：确保模型知道每个检测结果对应的类别。

    # 处理调色板信息（用于分割任务的可视化）
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    # 原理：将调色板信息添加到模型中，用于分割结果的可视化。
    # 目的：方便对分割结果进行可视化展示。

    # 根据是否使用分布式训练，选择单 GPU 测试或多 GPU 测试
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)
    # 原理：在单 GPU 环境下使用 MMDataParallel 包装模型进行单 GPU 测试，在多 GPU 环境下使用 MMDistributedDataParallel 包装模型进行多 GPU 测试。
    # 目的：根据不同的环境选择合适的测试方式，提高测试效率。

    # 获取当前进程的排名和总进程数
    rank, _ = get_dist_info()

    # 在主进程（rank=0）中处理结果
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            assert False
            #mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        # 如果指定了 --format_only 参数，则格式化结果
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        # 如果指定了 --eval 参数，则进行评估
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # 移除不必要的评估参数
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(eval_kwargs)
            print(dataset.evaluate(outputs, **eval_kwargs))
    # 原理：在主进程中处理结果，包括保存结果、格式化结果和评估结果。
    # 目的：将测试结果进行保存、格式化和评估，方便后续分析和展示。


if __name__ == '__main__':
    main()
