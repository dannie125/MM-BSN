# MM-BSN: Self-Supervised Image Denoising for Real-World with Mutil-Mask based on Blind-Spot Network
[MM-BSN(arxiv version)](https://arxiv.org/abs/2304.01598) has been accepted by CVPRw 2023.
---

![masks](./figs/big_f_try2.png)
## Abstract
Recent advances in deep learning have been pushing image denoising techniques to a new level. In self-supervised image denoising, blind-spot network (BSN) is one of the most common methods. However, most of the existing BSN algorithms use a dot-based central mask, which is recognized as inefficient for images with large-scale spatially correlated noise. In this paper, we give the definition of large-noise and propose a multi-mask strategy using multiple convolutional kernels masked in different shapes to further break the noise spatial correlation. Furthermore, we propose a novel self-supervised image denoising method that combines the multi-mask strategy with BSN (MM-BSN). We show that different masks can cause significant performance differences, and the proposed MM-BSN can efficiently fuse the features extracted by multi-masked layers, while recovering the texture structures destroyed by multi-masking and information transmission. Our MM-BSN can be used to address the problem of large-noise denoising, which cannot be efficiently handled by other BSN methods. Extensive experiments on public real-world datasets demonstrate that the proposed MM-BSN achieves state-of-the-art performance among self-supervised and even unpaired image denoising methods for sRGB images denoising, without any labelling effort or prior knowledge.

## Parameters
|   __Models__   |                                         __SIDD Validation__                                         |__Parameters__ |
|:----------:|:-----------------------------------------------------------------------------------------------:|:-------:|
| AP-BSN |                                             35.91/0.870                                             |   3.7M   |
| MM-BSN |                                             37.38/0.882                                             |   5.3M   |

## Setup
### Requirements

Our experiments are done with:

- Python 3.8.0
- PyTorch 1.12.0
- numpy 1.22.3
- opencv 4.6.0.66
- scikit-image 0.19.3

### Directory

Follow below descriptions to build code directory.

```
AP-BSN
├─ ckpt
├─ config
├─ DataDeal
├─ dataset
│  ├─ DND
│  ├─ SIDD
│  ├─ prep
│  ├─ test_data
├─ figs  
├─ model
├─ output
├─ util
```

- Make `dataset` folder which contains various dataset.
  - place [DND](https://noise.visinf.tu-darmstadt.de/), [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/) datasets at here.
  - `prep` folder contains prepared data for efficient training. (cropped patches with overlapping by `Data_prep.py`)
  - `test_data` folder contains some images with noise.
- Make `output` folder which contains experimental results including checkpoint, val/test images and training logs.
- Recommend to use __soft link__ due to folders would take large storage capacity.


## Quick test

To test noisy images with pre-trained MM-BSN in gpu:0.

```
python test.py -c SIDD -g 0 --pretrained ./ckpt/SIDD_MMBSN_o_a45.pth --test_dir ./dataset/test_data
```

---

## Training

```
usage: python train.py [-c CONFIG_NAME] [-g GPU_NUM] 
                       [-r RESUME] [-p PRETRAINED_CKPT] 
                       [-t THREAD_NUM] [-se SELF_ENSEMBLE]
                       [-sd OUTPUT_SAVE_DIR] [-rd DATA_ROOT_DIR]

Train model.

part of Arguments in config SIDD.yaml:  
   model:
  kwargs:
    type: MMBSN    # basic model types, eg.MMBSN, CSCBSN
    pd_a: 5
    pd_b: 2
    pd_pad: 2
    R3: True
    R3_T: 8
    R3_p: 0.16
    in_ch: 3
    bsn_base_ch: 128
    bsn_num_module: 9
    DCL1_num: 2
    DCL2_num: 7
    mask_type: 'o_fsz'  # mask types, eg. 'o_a45' means the combination of 'o' and 'a45'
    
```

Also, you can control other detail experimental configurations (e.g. training loss, epoch, batch_size, etc.) in each of config file.

## Evaluations
![masks](./figs/re.png)


## Acknowledgement
 Part of our codes are adapted from [AP-BSN](https://github.com/wooseoklee4/AP-BSN) and we are expressing gratitude for their work sharing.
 
## Cite
@misc{zhang2023mmbsn,
      title={MM-BSN: Self-Supervised Image Denoising for Real-World with Multi-Mask based on Blind-Spot Network}, 
      author={Dan Zhang and Fangfang Zhou and Yuwen Jiang and Zhengming Fu},
      year={2023},
      eprint={2304.01598},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
