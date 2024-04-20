<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">SplaTAM-Anything: Splat, track and map with any camera</h1>


<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
    <li>
      <a href="#downloads">Downloads</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## Installation

##### (Recommended)
SplaTAM has been tested on python 3.10, CUDA>=11.8. The simplest way to install all dependences is to use [anaconda](https://www.anaconda.com/) and [pip](https://pypi.org/project/pip/) in the following steps: 

```bash
conda create -n splatam python=3.10
conda activate splatam
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.7 -c pytorch -c conda-forge
pip install -r requirements.txt
```

<!-- Alternatively, we also provide a conda environment.yml file :
```bash
conda env create -f environment.yml
conda activate splatam
``` -->

## Usage

We will use the Replica dataset as an example to show how to prepare the dataset and use SplaTAM for sparse reconstruction. The following steps are similar for other datasets if you would like to test on. We ran our code on 2 Datasets - Replica and Scannetv1. Replica is an opensource dataset which can be downloaded by following the link in datasets section and for Scannet you would need to fill out a form to get the download link from the .

To run SplaTAM, please use the following command:

```bash
python scripts/splatam.py configs/iphone/splatam.py
```

To visualize the final interactive SplaTAM reconstruction, please use the following command:

```bash
python viz_scripts/final_recon.py configs/iphone/splatam.py
```

To visualize the SplaTAM reconstruction in an online fashion, please use the following command:

```bash
python viz_scripts/online_recon.py configs/iphone/splatam.py
```

To export the splats to a .ply file, please use the following command:

```bash
python scripts/export_ply.py configs/iphone/splatam.py
```

`PLY` format Splats can be visualized in viewers such as [SuperSplat](https://playcanvas.com/supersplat/editor) & [PolyCam](https://poly.cam/tools/gaussian-splatting).


## Downloads

DATAROOT is `./data` by default. Please change the `input_folder` path in the scene-specific config files if datasets are stored somewhere else on your machine.

### Replica

Download the data as below, and the data is saved into the `./data/Replica` folder. Note that the Replica data is generated by the authors of iMAP (but hosted by the authors of NICE-SLAM). Please cite iMAP if you use the data.

```bash
bash bash_scripts/download_replica.sh
```

### ScanNet

Please follow the data downloading procedure on the [ScanNet](http://www.scan-net.org/) website, and extract color/depth frames from the `.sens` file using this [code](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

<details>
  <summary>[Directory structure of ScanNet (click to expand)]</summary>

```
  DATAROOT
  └── scannet
        └── scene0000_00
            └── frames
                ├── color
                │   ├── 0.jpg
                │   ├── 1.jpg
                │   ├── ...
                │   └── ...
                ├── depth
                │   ├── 0.png
                │   ├── 1.png
                │   ├── ...
                │   └── ...
                ├── intrinsic
                └── pose
                    ├── 0.txt
                    ├── 1.txt
                    ├── ...
                    └── ...
```
</details>


We use the following sequences: 
```
scene0000_00
scene0059_00
scene0106_00
scene0181_00
scene0207_00
```

## Benchmarking

For running SplaTAM, They recommend running on weights and biases but it's okay to run offline for convinience. But if you still want wandb to log all your parameter, you can set the `wandb` flag to True in the configs file. Also make sure to specify the path `wandb_folder`. 

### Replica

To run SplaTAM on the `room0` scene, run the following command:

```bash
python scripts/splatam.py configs/replica/splatam.py
```

To run SplaTAM-S on the `room0` scene, run the following command:

```bash
python scripts/splatam.py configs/replica/splatam_s.py
```

For other scenes, please modify the `configs/replica/splatam.py` file or use `configs/replica/replica.bash`.



For other scenes, please modify the `configs/tum/splatam.py` file or use `configs/tum/tum.bash`.

### ScanNet

To run SplaTAM on the `scene0000_00` scene, run the following command:

```bash
python scripts/splatam.py configs/scannet/splatam.py
```

For other scenes, please modify the `configs/scannet/splatam.py` file or use `configs/scannet/scannet.bash`.


For other scenes, please modify the `configs/scannetpp/splatam.py` file or use `configs/scannetpp/scannetpp.bash`.


For other scenes, please modify the config files.

## Citation

If you find our paper and code useful, please cite us:

```bib
@inproceedings{keetha2024splatam,
        title={SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM},
        author={Keetha, Nikhil and Karhade, Jay and Jatavallabhula, Krishna Murthy and Yang, Gengshan and Scherer, Sebastian and Ramanan, Deva and Luiten, Jonathon},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2024}
      }
```

