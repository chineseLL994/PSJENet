# Joint-Pixel--and-Structure-Level-Enhancement-for-RGB-D-Video-Salient-Object-Detection
The codes of Joint Pixel- and Structure-Level Enhancement for RGB-D Video Salient Object Detection

🏃 The code is on the way...... 🏃 Our code will be publicly released after the paper is accepted.

# Requirements
* Python 3.10
* Torch 2.6.0
* Torchvision 0.21.0
* Cuda 12.6

# Usage
## To train
### Prepare RGB-D VSOD benchmarks
Download the datasets (![RDVS](https://github.com/kerenfu/RDVS), ![DVisal](https://github.com/DVSOD/DVSOD-DViSal) and ![ViDSOD](https://github.com/DVSOD/DVSOD-DViSal).

Please download the datasets from their official sources linked in the table above. Extract them into a single parent directory (e.g., /data). Your folder structure should look like this:
>/data/
>>── DViSal_dataset/
>>>── data/
>>>
>>>── test_all.txt
>>>>
>>── RDVS/
>>
>>>── test/
>>>
>>>── train/
>>>
>>── VidSOD/
>>
>>>── test/
>>>
>>>── train/
    

### Pretrain checkpoint download
Dependent Models: sam2_hiera_large.pt (from [facebookresearch/sam2](https://github.com/facebookresearch/sam2))


### Train and test
Use train.py and test.py to train and inference separately.
