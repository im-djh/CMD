# CMD: A Cross Mechanism Domain Adaptation Dataset for 3D Object Detection(ECCV2024)

An multi-mechanism, multi-modal real-world 3D object detection dataset that includes low-resolustion (32 beams) mechanical LiDAR, high-resolustion (128 beams) mechanical LiDAR, solid-state LiDAR, 4D millimeter-wave radar, and cameras. Each sensor is precisely time-synchronized and calibrated, making the dataset suitable for 3D object detection research involving multi-mechanism LiDAR data, particularly for cross-mechanism domain adaptation.

## Download
Option1: Log in [here](http://39.98.109.195:1000/) using the username "Guest" and the password "guest_CMD" to download the dataset.

Option2: Download CMD from [Hugging Face](https://huggingface.co/datasets/jinhaodeng/CMD/tree/main). 

## :balloon: CMD Cross-Mechanism Domain Adaptation 3D Object Detection Challenge

link: [Challenge](https://www.codabench.org/competitions/7749/)

### **In order to hold the challenge, we temporarily hid M1's training set data, which will be recovered after the competition.**

Please refer to the [documentation](docs/competitiontrack.md) for detailed steps

## Data Sample
![sample](docs/data_vis.png)

## Get Started

### 1. Installation and Data Preparation
**A.** Clone this repository.
```shell
git clone https://github.com/im-djh/CMD.git
```
**B.** Create virtual-env.
```shell
conda create -n xmuda python=3.8
```

**C.** Install requirements
cuda-11.4、cuda-11.6、cuda-11.7 tested
```
conda activate xmuda
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install spconv-cu116	
pip install -r requirements.txt
python setup.py develop
```

**D.** Download the dataset and create dataset infos

```
ln -s <path-to-the-downloaded-dataset> /xmuda/data/xmu
```
All the file will be organized as,
```
CMD
├── data
│   ├── xmu
│   │   │── ImageSets
|   |   |── label
|   |   |── seq**     
├── pcdet
├── tools
```

- Generate the data infos by running the following command: 
```python 
 python -m pcdet.datasets.xmu.xmu_dataset --func create_xmu_infos  --cfg_file tools/cfgs/dataset_configs/xmu/xmuda_dataset.yaml
```
- Generate gt_sampling_database by running the following command: 
```
python -m pcdet.datasets.xmu.xmu_dataset --func create_groundtruth_database  --cfg_file tools/cfgs/dataset_configs/xmu/xmu_dataset.yaml
```

**E.** For further steps, please refer to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

# Experimental Results
All LiDAR-based models are trained with 4 GTX 3090 GPU. 
Due to slight differences in annotation and calculation rules, there may be minor discrepancies between the experimental results and those reported in the paper.
## Model Zoo 
### 3D Object Detection Baselines
Selected supported methods are shown in the below table. The results are the 3D detection performance on the val set of our CMD.

#### Ouster
|AP@50                                                        | Car|Truck |Pedestrian | Cyclist | mAP    |
| ------------------------------------------------------ | :-----: | :--------: | :----: |:----: |:----: |
| [PointPillar](tools/cfgs/xmu_ouster_models/pointpillar_1x.yaml) | 41.70   | 18.13      | 3.80   | 37.77  |25.35|
[CenterPoint](tools/cfgs/xmu_ouster_models/centerpoint.yaml)| 40.43|18.77|11.47|45.76|29.11| 
[Voxel-RCNN](tools/cfgs/xmu_ouster_models/voxel_rcnn.yaml)| 43.20   | 21.70      | 13.70   | 41.32  |29.98| 
[VoxelNeXt](tools/cfgs/xmu_ouster_models/voxelnext_ioubranch_large.yaml) | 41.40   | 20.98      | 10.25   | 46.14  |29.70 |


#### Robosense
|AP@50                                                        | Car|Truck |Pedestrian | Cyclist | mAP    |
| ------------------------------------------------------ | :-----: | :--------: | :----: |:----: |:----: |
| [PointPillar](tools/cfgs/xmu_robosense_models/pointpillar_1x.yaml) | 47.63   | 18.83      | 6.82   | 36.98  |27.56|
[CenterPoint](tools/cfgs/xmu_robosense_models/centerpoint.yaml)| 49.16|21.21|2.79|44.82|29.50| 
[Voxel-RCNN](tools/cfgs/xmu_robosense_models/voxel_rcnn.yaml)| 50.61   | 23.97      | 12.86   | 43.17  |32.65| 
[VoxelNeXt](tools/cfgs/xmu_robosense_models/voxelnext_ioubranch_large.yaml) | 49.56   | 21.66      | 5.64   | 44.45  |30.33 |


#### Hesai
|AP@50                                                        | Car|Truck |Pedestrian | Cyclist | mAP    |
| ------------------------------------------------------ | :-----: | :--------: | :----: |:----: |:----: |
| [PointPillar](tools/cfgs/xmu_hesai_models/pointpillar_1x.yaml) | 42.11   | 18.85      | 6.89   | 33.27  |25.28|
[CenterPoint](tools/cfgs/xmu_hesai_models/centerpoint.yaml)| 42.39|19.15|4.02|37.88|25.86| 
[Voxel-RCNN](tools/cfgs/xmu_hesai_models/voxel_rcnn.yaml)| 44.85   | 21.84      | 11.63   | 34.81  |28.28| 
[VoxelNeXt](tools/cfgs/xmu_hesai_models/voxelnext_ioubranch_large.yaml) | 44.19   | 21.57      | 3.66   | 39.47  |27.22 |



## Training
```
cd ../../tools
```
- for single gpu
```
python train.py --cfg_file  cfgs/xmu_ouster_models/centerpoint.yaml 
```
- for multiple gpus (e.g. 8)
```
bash scripts/dist_train.sh 8 --cfg_file cfgs/xmu_ouster_models/centerpoint.yaml 
```

## Evaluation
- for single gpu
```
python test.py --cfg_file cfgs/xmu_ouster_models/centerpoint.yaml --ckpt /path/to/your/checkpoint 
```
- for multiple gpus (e.g. 8)
```
bash scripts/dist_test.sh 8 --cfg_file cfgs/xmu_ouster_models/centerpoint.yaml --ckpt /path/to/your/checkpoint 
```


## Todo List
- [ ] Make use of 4D Radar.
- [ ] Make use of Camera.

## Notes
- This reposity is developed based on OpenPCDet.
- Thanks to [Wei Ye](https://github.com/wayyeah) for his important contributions to the code repository.
- More about our lab can be found [here](https://asc.xmu.edu.cn/).

## Citation
If you find our **C**ross **M**echanism **D**ataset useful in your research, please consider cite:

```
@inproceedings{dengcmd,
  title={CMD: A Cross Mechanism Domain Adaptation Dataset for 3D Object Detection},
  author={Deng, Jinhao and Ye, Wei and Wu, Hai and Huang, Xun and Xia, Qiming and Li, Xin and Fang, Jin and Li, Wei and Wen, Chenglu and Wang, Cheng},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
