Pytorch implementation of Spatial Decomposition Net (SDNet) as described in 

> Chartsias, Agisilaos, et al. "Disentangled representation learning in cardiac image analysis." 
> Medical image analysis 58 (2019): 101535.

TODO:
- Add adversarial training

Requirements:
Pytorch 1.1, openCV, scipy, scikit-image, nibabel, bunch, tqdm

Usage:

1. Save data as 3D nifti files
2. Create a csv file listing the filenames of the training and validation images and ground 
truth (if applicable). See examples in `data_files/acdc_train_list_5subj.csv`.
2. Update parameters and file paths in `configs/vanilla_sdnet.json`
3. Run `python main.py --config=configs/vanilla_sdnet.json`

