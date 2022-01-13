# APE-V: Athlete Performance Evaluation using Video

The repository contains Code and Data pertaining to the paper "[APE-V: Athlete Performance Evaluation using Video](https://openaccess.thecvf.com/content/WACV2022W/CV4WS/html/Roygaga_APE-V_Athlete_Performance_Evaluation_Using_Video_WACVW_2022_paper.html)". 

- Files used for training and evaluating models with respect to experimental results presented in the paper.
- Code files common to all sub-experiments are present in their respective experiment folder [For e.g., "Code/Exp1__OpenPose_skeleton_LSTM" contains files pertaining to all Pose experiments], while those specific to an experiment (specifically related to Combined View models) are provided in their sub-folder.
- Experiment 1:
  - "Data/Skeleton data" contains skeleton points of the joints used for this experiment.
  - Each sub-folder corresponds to data for a different experiment. For e.g., "Data/Skeleton data/Confident Frames" corresponds to experiments on Confident Frames, with each corresponding file for the Center, Left, or Right view videos.
  - Each folder also contains additional data, corresponding to the Threshold experiment.
- Experiments 2 & 3:
  - Our current paths for videos are present in the csv files used for extracting and preprocessing the data for the Video Frame experiments. These should be replaced with the corresponding file paths at the time of experimentation.
  - The videos can be requested for use from the following link: https://cvrl.nd.edu/projects/data/ -> ND-Jump Analysis Video Dataset
- Each experiment sub-folder contains the hyperparameters file, which can be used to re-create the models presented in the paper. 

### Citation

```
@InProceedings{Roygaga_2022_WACV,
    author    = {Roygaga, Chaitanya and Patil, Dhruva and Boyle, Michael and Pickard, William and Reiser, Raoul and Bharati, Aparna and Blanchard, Nathaniel},
    title     = {APE-V: Athlete Performance Evaluation Using Video},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2022},
    pages     = {691-700}
}
```
