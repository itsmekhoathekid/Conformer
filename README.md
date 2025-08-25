# Conformer variants — My Implementation

> Implementation of **Conformer** and **Multi-Convformer** for end-to-end speech recognition:
> 📄 Conformer : https://arxiv.org/pdf/2005.08100
> 📄 Multi-ConvFormer: https://arxiv.org/pdf/2407.03718

## 🔗 Repository
- GitHub: https://github.com/itsmekhoathekid/Conformer

## 🚀 Quickstart

### 1) Clone & Setup
```
bash
git clone https://github.com/itsmekhoathekid/Conformer
cd Conformer
```

### 2) Download & Prepare Dataset
This will download the datasets configured inside the script and generate manifests/features as needed.
```
bash
bash ./prep_data.sh
```

### 3) Train
Train with a YAML/JSON config of your choice.
```
bash
python train.py --config path/to/train_config.yaml
```

### 4) Inference (example)
```
bash
python infererence.py --config path/to/train_config.yaml --epoch num_epoch
```

## 📦 Project Layout (typical)
```
Conformer/
├── prep_data.sh                 # dataset download & preprocessing
├── train.py                     # training entry point
├── inference.py                     # inference script (optional)
├── configs/                     # training configs (yaml/json)
├── models/                    # model, losses, data, utils
│   ├── model.py
│   ├── encoder.py
│   ├── decoder.py
│   └── ...
├── utils/ 
└── README.md
```

