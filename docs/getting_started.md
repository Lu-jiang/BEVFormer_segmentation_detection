# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train BEVFormer with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 8
```

Eval BEVFormer with 8 GPUs
```
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./path/to/ckpts.pth 8
```
Note: using 1 GPU to eval can obtain slightly higher performance because continuous video may be truncated with multiple GPUs. By default we report the score evaled with 8 GPUs.



# Using FP16 to train the model.
The above training script can not support FP16 training, 
and we provide another script to train BEVFormer with FP16.

```
./tools/fp16/dist_train.sh ./projects/configs/bevformer_fp16/bevformer_tiny_fp16.py 8
```


# Visualization 

see [visual.py](../tools/analysis_tools/visual.py)

# Visual
## 运行debug_test.py (you should floow me.)

'''
python debug_test.py --config $config_path --checkpoint $ckpt_path --show-dir $seg_result_save_path
'''

## 运行visual_det_seg.py

'''
python visual_det_seg.py
注意更改该python文件中的pred_seg_path =$seg_result_save_path
'''
