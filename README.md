## ExeChecker
This repo holds the code for: ExeChecker: Where did I go wrong? ([pdf]())

## Data Preparation 
We experimented on two datasts: **ExeCheck** and **UI-PRMD**. Both datasets contains correct and incorrect movements related to common exercises performed by patients in physical therapy and rehabilitation programs. To use for our experiments, you can unzip the files in `processed_execheck` and `processed_uiprmd` folders and proceed with the training.
The processed dataset has been segmented on repititions and augmented with mirroring.
### ExeCheck
The [ExeCheck](https://www.cs.bu.edu/faculty/betke/ExeChecker/) dataset consists of RGB-D videos of 10 rehabilitation exercises performed by 7 healthy subjects collected by a Azure Kinect sensor. Each exercise has a paired performance in both correct and incorrect forms by the same subject with 5 movement repetitions. You can download the original dataset from the [Dropbox](https://www.dropbox.com/scl/fo/lnqjvz4iiodew7aw5ur7m/AG8WUBHomA2EdgNmTmits2g?rlkey=l9pmf59h9eegsnvacrjw4tn7h&st=77lmqg7w&dl=0)
### UI-PRMD
The [UI-PRMD](https://webpages.uidaho.edu/ui-prmd/) consists of 10 rehabilitation movements. A sample of 10 healthy individuals repeated each movement 10 times in front of two sensory systems for motion capturing: a Vicon optical tracker, and a Kinect camera. The data is presented as positions and angles of the body joints in the skeletal models provided by the Vicon and Kinect mocap systems. 
### Custom Dataset
You can also create your own dataset using the scripts in the `prepare` folder with cooresponding modifications.


## Training & Testing
Change the config file according to your needs, then run with 
`python trainMulti_perExe.py --config ./config/execheck_Multi_perExe.yaml`


## Citation
If you find our research helpful, please consider cite this work:
```
placeholder
```

## Code Acknowledgment
 - We use [STGAT](https://github.com/hulianyuyy/STGAT) as our feature extractor. Thanks to the original authors for their work!
