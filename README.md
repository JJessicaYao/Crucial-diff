# Crucial-Diff: A Unified Diffusion Model for Crucial Image and Annotation Synthesis in Data-scarce Scenarios
Official Code for paper "Crucial-Diff: A Unified Diffusion Model for Crucial Image and Annotation Synthesis in Data-scarce Scenarios"

> The scarcity of data in various scenarios, such as medical, industry and autonomous driving, leads to model overfitting and dataset imbalance, thus hindering effective detection and segmentation performance. Existing studies employ the generative models to synthesize more training samples to mitigate data scarcity. However, these synthetic samples are repetitive or simplistic and fail to provide ``crucial information" that targets the downstream model's weaknesses. Additionally, these methods typically require separate training for different objects, leading to computational inefficiencies. To address these issues, we propose Crucial-Diff, a domain-agnostic framework designed to synthesize crucial samples. Our method integrates two key modules. The Scene Agnostic Feature Extractor (SAFE) utilizes a unified feature extractor to capture target information. The Weakness Aware Sample Miner (WASM) generates hard-to-detect samples using feedback from the detection results of downstream model, which is then fused with the output of SAFE module. Together, our Crucial-Diff framework generates diverse, high-quality training data, achieving a pixel-level AP of 83.63\% and an F1-MAX of 78.12\% on MVTec. On polyp dataset, Crucial-Diff reaches an mIoU of 81.64\% and an mDice of 87.69\%.

## The code will be release soon.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crucial-diff.git
cd crucial-diff

# Install dependencies
conda env create cruicaldiff
conda activate cruicaldiff
pip install -r requirements.txt

# Install Accelerate for distributed training
pip install accelerate
accelerate config  # Configure your distributed training setup
```

## Training
###  Stage 1: Training SAFE module 
```bash
# Option 1: Using shell script
sh scripts/train_baseline.sh

# Option 2: Direct command line execution
accelerate launch train_baseline.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --placeholder_token="S" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=1000 \
  --learning_rate=1e-06 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$BASELINE_SAVE_DIR \
  --object="all" \
  --save_steps 1000 
```

###  Stage 2: Training downstream segmentation model
This stage implements a downstream segmentation model based on U-Net architecture, following the approach used in [AnomalyDiffusion](https://github.com/sjtuplayer/anomalydiffusion/tree/master). 
> Note: This U-Net implementation is provided as an example. You can replace it with any segmentation model of your choice.
```bash
python train-localization.py --generated_data_path $path_to_mvtec_train_set --mvtec_path=$path_to_mvtec
```

###  Stage 3: Training WASM module 
```bash
# Option 1: Using shell script
sh scripts/train_baseline.sh

# Option 2: Direct command line execution
accelerate launch train_contrast.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --placeholder_token="S" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=200 \
  --learning_rate=1e-06 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$SAVE_DIR \
  --object_mapper_path=$BASELINE_SAVE_DIR \
  --object="all" \
  --seg_checkpoint_path=$DOWNSTREAM_MODEL_CKPT_DIR \
  --save_steps 1000
```
