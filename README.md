<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">SplaTAM++: Full stack splatting based SLAM</h1>


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

We will use the Replica dataset as an example to show how to prepare the dataset and use SplaTAM for sparse reconstruction. The following steps are similar for other datasets if you would like to test on. We ran our code on 2 Datasets - Replica and Scannetv1. Replica is an opensource dataset which can be downloaded by following the link in datasets section and for Scannet you would need to fill out a form to get the download link from the official developers.

To run SplaTAM, please use the following command:

```bash
python scripts/splatam.py configs/replica/splatam.py
```
To run SplaTAM-S on the `room0` scene, run the following command:

```bash
python scripts/splatam.py configs/replica/splatam_s.py
```

To visualize the final interactive SplaTAM reconstruction, please use the following command:

```bash
python viz_scripts/final_recon.py configs/replica/splatam.py
```

To visualize the SplaTAM reconstruction in an online fashion, please use the following command:

```bash
python viz_scripts/online_recon.py configs/replica/splatam.py
```

To export the splats to a .ply file, please use the following command:

```bash
python scripts/export_ply.py configs/replica/splatam.py
```



## Downloads

DATAROOT is `./data` by default. Please change the `input_folder` path in the scene-specific config filees. you are free to change this depending on your location. The model link is also given after the dataset download.

### Replica

Download the data as below, and the data is saved into the `./data/Replica` folder.

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
```
### Model Download link

Below is the large version of the Depth-Anything model you can find the small and medium versions in te same source.

(https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth)

# Data sampling and conversion

For imitating sparse data from the Lidar sensors, first install the  file for Depth Anything from below. Keep the  file in the main directory, Keep in mind the default data folder is data/..... for Replica and Scannet. the conversion code read from there but you can also provide a argument for input path to do the same thing. After Creating the converted files replace the new depth files wth the original andcontinue to run the splatam.

```bash
python scripts/data_prep.py --img-path data/Replica/room0/results --outdir ./converted_depth --grayscale
python scripts/data_prep.py --img-path <path-to-input-directory> --outdir <path-to-output-directory --grayscale

```

For running SplaTAM, The original oauthors recommend running on weights and biases but it's okay to run offline for convinience. But if you still want to run on wandb and log all your parameter, you can set the `wandb` flag to True in the configs file. Also make sure to specify the path `wandb_folder`. 


For other scenes, please modify the `configs/replica/splatam.py` file or use `configs/replica/replica.bash`.


### ScanNet

To run SplaTAM on the `scene0000_00` scene, run the following command:

```bash
python scripts/splatam.py configs/scannet/splatam.py
```

For other scenes, please modify the `configs/scannet/splatam.py` file or use `configs/scannet/scannet.bash`.

###

