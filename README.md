# Efficient Wavelet ViT

This repository is the official implementation of the paper "_Combining Efficient-ViT with Wavelet Transform: An Attention-based Structure for Face Forgery Detection_". 

The code is based on the Efficient-ViT model and incorporates multi-level wavelet transform to enhance the feature extraction process. By devising an adaptive attention mechanism, the model can effectively integrate the frequency features with spatial features. 

Using this repository, you can train and evaluate the Efficient Wavelet ViT model for two public datasets, Celeb-DF and FaceForensics++, for deepfake detection tasks.

## Setup

Clone this repository and install the required packages:

```bash
git clone https://github.com/Sheldon-Xiao9/efficient-wavelet-vit.git

cd efficient-wavelet-vit

pip install -r requirements.txt
```

Remember to install the `torch` and `torchvision` versions that are compatible with your CUDA version. You can find the installation command for your specific setup [here](https://pytorch.org/get-started/locally/). We also recommend using `python 3.12` or later for better compatibility with our code, as we trained and tested the model on this version.

## Dataset Preparation

Download and extract the datasets from the following links:

- [FaceForensics++](https://github.com/ondyari/FaceForensics/tree/master/dataset)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)

After downloading, please use our script to convert the videos to frames. You can use the following command to convert the videos to frames:

```bash
python data/FaceForensics++/extract_compressed_videos.py \
    --data_path /path/to/ff++/videos \
    --dataset all \
    --output_path /path/to/output/frames

python data/Celeb-DF-v2/extract_frames.py 
    --data_path /path/to/celebdf/videos \
    --category all \
    --output_path /path/to/output/frames \
    --testing_file /path/to/test/splits
```

You can choose to extract all videos or only the ones you need. The `--dataset` argument in FaceForensics++ can be set to `all`, `Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, `NeuralTextures`, or `original`. On the contrary, the `--category` argument in Celeb-DF can be set to `train` or `test`, and the path to the test split file is required if you choose `test`, where you need to specify it in the `--testing_file` argument.

After extracting the frames, place the datasets in the following directory structure. For FaceForensics++, the dataset should be organized as follows:

```
faceforensics/
└── ff++/
    ├── frames/
    │   ├── Deepfakes/
    |   |   ├── 001_002/
    |   |   |   ├── 0001.png
    |   |   |   └── ...
    |   |   ├── 002_003/
    |   |   └── ...
    |   ├── Face2Face/
    |   |   └── ...
    |   ├── FaceShifter
    |   |   └── ...
    │   ├── FaceSwap
    │   |   └── ...
    │   ├── NeuralTextures
    |   |   └── ...
    │   └── original
    │       ├── 001/
    │       |   ├── 0001.png
    │       |   └── ...
    │       └── ...
    └──splits/
        ├── train.json
        ├── val.json
        └── test.json
```

For Celeb-DF, the dataset should be organized as follows:

```
celebdf/
└── frames/
    ├── Celeb-real/
    │   ├── id0_0000/
    │   |   ├── 0001.png
    │   |   └── ...
    │   ├── id0_0001/
    │   └── ...
    └── Celeb-synthesis/
        ├── id0_id16_0000/
        |   ├── 0001.png
        |   └── ...
        ├── id0_id16_0001/
        └── ...
```

We suggest using the `ff++` and `celebdf` folders as the root directory for the datasets. You can modify the dataset paths in the `config/data_loader.py` file to match your directory structure.

## Training

To train the Efficient Wavelet ViT model, you can use the following command:

```bash
python train.py --root /path/to/dataset
```

By default the command will train the model on the FaceForensics++ dataset. You can also customize the following training parameters:

- `--output`: Path to save the model checkpoints and best model (default: `./output`)
- `--batch-size`: Batch size for training (default: 8)
- `--epochs`: Number of epochs for training (default: 30)
- `--lr`: Learning rate (default: 1e-4)
- `--dim`: Dimension of the features (default: 128)
- `--frame-count`: Number of frames to be used for training (default: 300)
- `--accum-steps`: Number of gradient accumulation steps (default: 2)
- `--seed`: Random seed for training (default: 42)
- `--visualize`: Whether to visualize the training process after the training
- `--multi-gpu`: Whether to use multiple GPUs for training 
- `--resume`: If you need to resume training from a checkpoint, please specify the path to the checkpoint file for this argument

We do not provide the arguments for Efficient-ViT configuration, as you can modify it in the `config/architecture.yaml` file. You can also modify the arguments in the `config/data_loader.py` and `config/transform.py` files to customize the data loader and data augmentation settings.

## Evaluation

To evaluate the Efficient Wavelet ViT model, you can use the following command:

```bash
python eval.py \
    --root /path/to/dataset \
    --model-path /path/to/model \
```

Like the training command, you can customize the following evaluation parameters as well:

- `--output`: Path to save the evaluation results (default: `./output/eval`)
- `--batch-size`: Batch size for evaluation (default: 8)
- `--frame-count`: Number of frames to be used for evaluation (default: 300)
- `--seed`: Random seed for evaluation (default: 42)
- `--dataset`: Dataset to be used for evaluation (default: `ff++`)
    - `ff++`: FaceForensics++
    - `celebdf`: Celeb-DF(V2)
- `--test-list`: If you choose to evaluate on Celeb-DF(V2), you'll need to provide the root directory of the test split file for this argument
- `--ablation`: IF the model is trained for ablation study, please choose which ablation mode to evaluate (default: `dynamic`)
    - `dynamic`: All modules activated
    - `sfe_only`: Only SFE module activated
    - `sfe_mwt`: SFE and MWT modules activated
- `--visualize`: Whether to visualize the evaluation process after the evaluation

We also provide other scripts addressed in our paper, including `plot_celebdf_roc.py` for plotting the ROC curve of cross-dataset evaluation on Celeb-DF, and `visualize_features_maps.py` for visualizing the feature maps of SFE and MWT modules. You can use the following command to run these scripts:

```bash
python plot_celebdf_roc.py --model-paths /path/to/model1 /path/to/model2

python visualize_feature_maps.py --model-path /path/to/model
```

These scripts are under the `utils` folder, and you can modify the parameters in the script of `plot_celebdf_roc.py` to customize the generation of ROC curves, including the labels, model types, and whether to evaluate on frame-level. For `visualize_feature_maps.py`, we also provide options for visualizing the feature maps of SFE and MWT modules. 

## Ablation

If you're interested in conducting ablation studies, we provide a script to help you with that. You can use the following command to run the ablation study:

```bash
python ablation.py --root /path/to/dataset 
```

This script will automatically run the ablation study for you, in a order of `sfe_only`, `sfe_mwt`, and `dynamic`. The following parameters are available for customization:

- `--output`: Path to save the ablation study results (default: `./output/ablation`)
- `--batch-size`: Batch size for ablation study (default: 8)
- `--epochs`: Number of epochs for ablation study (default: 30)
- `--lr`: Learning rate (default: 1e-4)
- `--dim`: Dimension of the features (default: 128)
- `--frame-count`: Number of frames to be used for ablation study (default: 300)
- `--seed`: Random seed for ablation study (default: 42)

## Acknowledgements
This repository is built upon the following works:

- [Efficient ViT](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection)
- [Multi-Scale Wavelet Transformer](https://arxiv.org/abs/2210.03899)
- [FaceForensics++](https://github.com/ondyari/FaceForensics/tree/master/dataset)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)

Special thanks to the authors of these works for their contributions to the field of deepfake detection.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

We will release the paper soon, once it is accepted. 