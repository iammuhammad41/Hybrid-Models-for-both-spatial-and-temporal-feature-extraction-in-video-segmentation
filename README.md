 # Hybrid Models for Spatial and Temporal Feature Extraction in Video Segmentation

This repository demonstrates three complementary approaches for segmentation tasks, combining modern architectures to capture spatial and temporal features:

1. **Semantic Segmentation using Vision Transformers (ViTs)**
2. **Vehicle Spatial-UNet Segmentation**
3. **Hybrid CNN–RNN Model for Video Segmentation**

Each method is provided as a Jupyter/Colab notebook in the `notebooks/` folder.


# Hybrid-Models-for-both-spatial-and-temporal-feature-extraction-in-video-segmentation
Hybrid models combine multiple neural network architectures to tackle complex tasks. 
- Semantic Segmentation Using ViTs
- Vehicle Spatial-UNet Segmentation
- a hybrid of CNNs and RNNs can be used for both spatial and temporal feature extraction in video segmentation
- longitudinal medical imaging tasks.


## 📁 Repository Structure

```
hybrid-video-segmentation/
├── notebooks/
│   ├── semantic_segmentation.ipynb       # ViT-based image/video segmentation
│   ├── vehicle_spatial_unet_segmentation.ipynb  # CNN-based spatial U-Net for vehicle scenes
│   └── hybrid_cnn_rnn_video_segmentation.ipynb  # CNN+RNN model for spatiotemporal segmentation
├── data/                                 # (Optional) Sample images or TFRecords
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```



## 🔧 Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Dependencies include:

* `torch`, `torchvision`, `transformers`, `timm`, `tensorflow`
* `efficientnet`, `tensorflow_addons`
* `segmentation_models_pytorch` (optional), `keras`, `scikit-image`
* `matplotlib`, `seaborn`, `pandas`, `numpy`
* `opencv-python`, `tqdm`, `Pillow`


## 🚀 Usage

Open the desired notebook in Jupyter or Colab and run all cells:

### 1. Semantic Segmentation with ViTs

* Uses `transformers` DetrForSegmentation or `openmmlab/upernet-convnext-base` for semantic masks.
* Loads an example COCO image and displays segmentation logits.

### 2. Vehicle Spatial-UNet Segmentation

* Reads CVPR-2018 autonomous driving dataset samples.
* Builds a binary mask for vehicle classes and trains a U‑Net with spatial attention.
* Visualizes input, mask, and overlay crops.

### 3. Hybrid CNN–RNN Video Segmentation (Longitudinal Medical Imaging)

* (Placeholder notebook) Combines a CNN encoder (for spatial features) and an RNN (ConvLSTM) decoder for temporal modeling.
* Processes video frames or longitudinal medical image sequences.

Each notebook contains data loading, model definition, training loops, evaluation metrics, and visualizations.


## 🛠️ Configuration & Customization

* **Data paths**: Update dataset directories at the top of each notebook.
* **Model hyperparameters**: Learning rates, batch sizes, epochs can be tuned in the notebooks.
* **Backbones**: Swap Vision Transformer, EfficientNet, or ResNet backbones as desired.
* **Loss functions**: Use Dice, BCE, or cross-entropy as appropriate.
* **Temporal module**: In the hybrid CNN–RNN, you can switch between ConvLSTM, LSTM, or Transformer layers.


