# Quick‑Start Tutorial

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
- Modify the sensor list in the same file (xmu_dataset.py, line 628):
```
sensors = ['ouster', 'hesai']
```
- Generate gt_sampling_database by running the following command: 
```
python -m pcdet.datasets.xmu.xmu_dataset --func create_groundtruth_database  --cfg_file tools/cfgs/dataset_configs/xmu/xmu_dataset.yaml
```
**E.** Switch to the Test‑Set Configuration

- Edit your model YAML and override the DATA_CONFIG block:
```
DATA_CONFIG:
  _BASE_CONFIG_: tools/cfgs/dataset_configs/xmu/xmu_dataset_robosense_test.yaml
```

**F.** Run Inference and Produce result.pkl
```
bash scripts/dist_test.sh 4 \
    --cfg_file  XXX \
    --ckpt      XXX \
```
**G.** Submit result.pkl to CodaBench