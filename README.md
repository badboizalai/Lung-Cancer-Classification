# Lung-Cancer-Classification

# Lung Cancer Classification with SAM Segmentation and Deep Learning

A research project applying deep learning models combined with SAM (Segment Anything Model) to classify lung CT scan images into three categories: normal, benign, and malignant. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

## 📋 Overview

This project compares the performance of 6 different deep learning architectures for lung cancer classification from CT scans: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **MobileNet** ⭐ (Highest accuracy: 94.46%)
- DenseNet121 (92.75%)
- VGG16 (85.32%)
- AlexNet (73.41%)
- Custom CNN (57.62%)
- ResNet50 (45.15%)

## ✨ Key Features

- **SAM Segmentation**: Utilizes Segment Anything Model for precise lung region segmentation [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **Data Augmentation**: Advanced augmentation techniques including rotation, flipping, Gaussian blur, and histogram equalization [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **Class Imbalance Handling**: Enhanced dataset from 110 cases to 6,000 images (2,000 per class) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **Multi-Model Comparison**: Comprehensive evaluation of 6 deep learning architectures [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

## 📊 Dataset

**IQ-OTH/NCCD Lung Cancer Dataset**
- **Original data**: 1,190 CT slices from 110 cases [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
  - 40 malignant cases
  - 15 benign cases
  - 55 normal cases
- **After augmentation**: 6,000 images (balanced across 3 classes) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **Data split**: 80% training / 20% testing [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/73ad15e7-45cd-4245-9a22-3ad8135e47de/Lung_Cancer_Model.ipynb)

## 🛠️ Installation

### System Requirements
```bash
Python 3.8+
CUDA (recommended for training)
```

### Install Dependencies

```bash
# Clone repository
git clone <repository-url>
cd lung-cancer-classification

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download SAM checkpoint
mkdir -p sam_checkpoints
wget -O sam_checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Install other dependencies
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn torch scipy scikit-image
```

## 🚀 Usage

### 1. Data Preparation
```python
import os
import pandas as pd

# Path to data directory
directory = "/content/Augmented_Data"
categories = ['Bengin cases', 'Malignant cases', 'Normal cases']

# Create dataframe
filepaths = []
labels = []

for category in categories:
    path = os.path.join(directory, category)
    if os.path.exists(path):
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            filepaths.append(filepath)
            labels.append(category)

# Create DataFrame
Lung_df = pd.DataFrame({"filepaths": filepaths, "labels": labels})
```

### 2. Training Models
```python
# Open and run the notebook
jupyter notebook Lung_Cancer_Model.ipynb

# Training configuration
INPUT_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 50
```

### 3. Model Evaluation
Models are evaluated using the following metrics: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 📈 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **MobileNet** | **94.46%** | **94.65%** | **94.46%** | **94.47%** |
| DenseNet121 | 92.75% | 92.82% | 92.75% | 92.77% |
| VGG16 | 85.32% | 85.57% | 85.32% | 85.32% |
| AlexNet | 73.41% | 74.88% | 73.41% | 73.12% |
| Custom CNN | 57.62% | 58.08% | 57.62% | 57.59% |
| ResNet50 | 45.15% | 45.16% | 45.15% | 44.34% |

*Source: * [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

### Key Findings

**MobileNet** achieved the best performance due to: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- Depthwise separable convolutions reduce overfitting
- Lightweight architecture suitable for small datasets
- Excellent integration with SAM segmentation masks
- Efficient feature extraction with minimal parameters

**ResNet50** underperformed because of: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- Gradient saturation in shallow layers
- Over-complex architecture for limited data
- Insufficient regularization for the dataset size

## 🔬 Methodology

### SAM Segmentation Pipeline [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
1. Use SAM to generate lung region masks
2. Place point prompts on bright regions in CT images
3. Combine original images with masked versions for diverse training data
4. Enhance tumor localization by focusing on relevant lung areas

### Data Augmentation Techniques [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **Rotation**: Up to 30 degrees to simulate different imaging orientations
- **Flipping**: Horizontal and vertical for more variability
- **Gaussian Blur**: Simulating varying levels of focus and image quality
- **Histogram Equalization**: Enhancing contrast in CT scans
- **Shear Transformations**: Simulating different imaging perspectives
- **Zoom**: Improving detection at different scales

### Training Configuration [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/73ad15e7-45cd-4245-9a22-3ad8135e47de/Lung_Cancer_Model.ipynb)
- **Input size**: 224×224 pixels
- **Optimizer**: Adam (learning rate: 0.0001)
- **Batch size**: 32
- **Epochs**: Up to 50 with early stopping
- **Data split**: 90% training / 10% validation [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

### Regularization Techniques [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **Early stopping**: Monitor validation loss
- **Learning rate reduction**: Dynamic adjustment on plateau
- **Dropout layers**: Prevent overfitting in fully connected layers

### Evaluation Metrics [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

```python
# Mathematical formulas
Accuracy = (TN + TP) / (TN + TP + FN + FP)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
Specificity = TN / (TN + FP)
```

Where:
- **TP (True Positive)**: Correctly predicted positive cases
- **TN (True Negative)**: Correctly predicted negative cases
- **FP (False Positive)**: Incorrectly predicted positive cases
- **FN (False Negative)**: Incorrectly predicted negative cases

## 📁 Project Structure

```
lung-cancer-classification/
├── Lung_Cancer_Model.ipynb          # Main notebook with full pipeline
├── conference_latex_template.pdf    # Research paper
├── sam_checkpoints/                 # SAM model checkpoints
│   └── sam_vit_b_01ec64.pth
├── Augmented_Data/                  # Augmented dataset
│   ├── Normal cases/                # 2,000 normal images
│   ├── Bengin cases/                # 2,000 benign images
│   └── Malignant cases/             # 2,000 malignant images
├── best_mobilenet_model.keras       # Best trained model
├── models/                          # Saved model checkpoints
│   ├── mobilenet/
│   ├── densenet121/
│   ├── vgg16/
│   ├── alexnet/
│   ├── resnet50/
│   └── custom_cnn/
└── README.md                        # This file
```

## 👥 Authors

**Ta Khoi Nguyen**  
Department of Artificial Intelligence  
FPT University  
FPT City, Da Nang, 550000, Vietnam  
Email: takhoinguyengl@gmail.com

**Thai Nhat Tan**  
Department of Artificial Intelligence  
FPT University  
FPT City, Da Nang, 550000, Vietnam  
Email: thainhattan26012003@gmail.com

*Source: * [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

## 🔮 Future Directions

1. **3D CNNs for Volumetric Analysis**: Transition from 2D slices to 3D models to capture spatial relationships across multiple slices and improve overall classification accuracy [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

2. **Federated Learning**: Train on multi-center data while preserving patient privacy and data security, making the model more robust and applicable to various healthcare settings [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

3. **Multi-Modality Integration**: Combine CT scans with MRI and PET imaging to provide complementary information for more comprehensive tumor classification [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

4. **Ensemble Learning**: Investigate combining multiple deep learning models in an ensemble framework to leverage the strengths of each architecture [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

5. **Real-Time Clinical Deployment**: Implement user-friendly interfaces for real-time diagnosis assistance, ensuring model explainability and clinical utility [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

## ⚠️ Limitations

- **Small Dataset**: Only 110 original cases despite augmentation to 6,000 images [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **SAM Dependency**: Relies on bright-region prompts, may miss subtle nodules or early-stage cancers with low contrast [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **Generalization Concerns**: Trained on single-center data; needs validation on real-world multi-center datasets with diverse patient demographics and imaging protocols [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **2D Analysis**: Current implementation uses 2D slices rather than full 3D volumetric data [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

## 📚 References

1. Shorten, C. and Khoshgoftaar, T.M., 2019. "A survey on image data augmentation for deep learning." *Journal of Big Data*, 6(1), p.60. DOI: 10.1186/s40537-019-0197-0 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

2. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.-Y. and Dollár, P., 2023. "Segment Anything." *arXiv preprint arXiv:2304.02643* [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

3. LeCun, Y., Bengio, Y. and Hinton, G., 2015. "Deep learning." *Nature*, 521(7553), pp.436-444. DOI: 10.1038/nature14539 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

4. Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M. and Adam, H., 2017. "MobileNets: Efficient convolutional neural networks for mobile vision applications." *arXiv preprint arXiv:1704.04861* [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

5. Simonyan, K. and Zisserman, A., 2014. "Very deep convolutional networks for large-scale image recognition." *arXiv preprint arXiv:1409.1556* [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

6. Huang, G., Liu, Z., Van Der Maaten, L. and Weinberger, K.Q., 2017. "Densely connected convolutional networks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp.4700-4708. DOI: 10.1109/CVPR.2017.634 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

7. He, K., Zhang, X., Ren, S. and Sun, J., 2016. "Deep residual learning for image recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp.770-778. DOI: 10.1109/CVPR.2016.90 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

8. Krizhevsky, A., Sutskever, I. and Hinton, G.E., 2012. "ImageNet classification with deep convolutional neural networks." *Advances in Neural Information Processing Systems (NeurIPS)*, pp.1097-1105 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

## 🎯 Key Contributions

- **Novel SAM Integration**: First study to combine SAM segmentation with deep learning for lung cancer classification on CT scans [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **Comprehensive Architecture Comparison**: Systematic evaluation of 6 CNN architectures on the same dataset [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **Effective Imbalance Handling**: Advanced augmentation techniques successfully balanced classes from 40:15:55 ratio to 2000:2000:2000 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- **MobileNet Superiority**: Demonstrated MobileNet's effectiveness for medical imaging with limited data, achieving 94.46% accuracy [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

## 📊 Technical Specifications

### Model Architectures [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

**MobileNet**
- Depthwise separable convolutions
- Efficient and lightweight design
- Ideal for real-time applications
- Small model size without sacrificing accuracy

**VGG16**
- 16-layer CNN with small receptive fields
- Deep stacks of convolutional layers
- Simple and effective for visual recognition
- Computationally intensive

**DenseNet121**
- Dense blocks with feature reuse
- Each layer receives input from all previous layers
- Enhanced gradient flow during training
- Improved model performance

**AlexNet**
- Traditional CNN architecture
- Local response normalization
- Large number of filters in early layers
- Pioneering deep learning model

**ResNet50**
- Residual learning with skip connections
- Deep architecture (50 layers)
- Captures complex features and patterns
- Popular for image classification

**Custom CNN**
- Baseline model: 2 convolutional layers + 2 fully connected layers
- Simple architecture for comparison
- Demonstrates benefits of advanced pre-trained models

## 🛡️ Clinical Significance

This research demonstrates the potential of deep learning techniques in revolutionizing lung cancer detection: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

- **Early Detection**: Improved accuracy in identifying malignant cases, crucial for patient survival rates
- **Radiologist Support**: Automated ROI highlighting assists in faster and more accurate diagnosis
- **Efficiency**: Reduces time-consuming manual interpretation of CT scans
- **Consistency**: Minimizes inter-observer variability in diagnosis
- **Clinical Decision Support**: Offers new avenues for reliable, automated diagnostic tools in medical imaging

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{nguyen2024lung,
  title={Enhancing Lung Cancer Classification with SAM Segmentation and Deep Learning on CT Scans},
  author={Nguyen, Ta Khoi and Tan, Thai Nhat},
  institution={FPT University, Department of Artificial Intelligence},
  address={Da Nang, Vietnam},
  year={2024}
}
```

## 📄 License

This project is released for research and educational purposes. Please add an appropriate license (MIT, Apache 2.0, GPL, etc.) based on your requirements.

## 🙏 Acknowledgments

- **IQ-OTH/NCCD Lung Cancer Dataset** for providing the foundational data
- **Meta AI Research** for the Segment Anything Model (SAM)
- **FPT University**, Department of Artificial Intelligence
- **Da Nang, Vietnam** research community

## 💻 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
- **RAM**: Minimum 16GB (32GB recommended)
- **Storage**: At least 10GB free space for dataset and models
- **CPU**: Multi-core processor (Intel i5/i7 or AMD equivalent)

### Software
- **Python**: 3.8 or higher
- **TensorFlow**: 2.x
- **PyTorch**: Latest version (for SAM)
- **CUDA Toolkit**: 11.x or 12.x (for GPU acceleration)
- **cuDNN**: Compatible version with CUDA

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Reduce batch size
BATCH_SIZE = 16  # Instead of 32
```

**2. SAM Model Loading Error**
```bash
# Verify checkpoint download
ls -lh sam_checkpoints/sam_vit_b_01ec64.pth
# Should be approximately 358MB
```

**3. Data Path Errors**
```python
# Verify directory structure
import os
print(os.listdir('Augmented_Data'))
# Should show: ['Normal cases', 'Bengin cases', 'Malignant cases']
```

**4. Import Errors**
```bash
# Reinstall dependencies
pip install --upgrade tensorflow torch torchvision
```

## 📞 Contact & Support

For questions, issues, or collaboration opportunities:
- **GitHub Issues**: Open an issue on the repository
- **Email**: takhoinguyengl@gmail.com
- **Institution**: FPT University, Department of Artificial Intelligence

## 🌟 Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd lung-cancer-classification
pip install -r requirements.txt

# 2. Download SAM checkpoint
bash download_sam.sh

# 3. Prepare data
python prepare_data.py --data_dir /path/to/data

# 4. Train model
python train.py --model mobilenet --epochs 50

# 5. Evaluate
python evaluate.py --model_path best_mobilenet_model.keras
```

## 📈 Performance Visualization

The project includes visualization tools for:
- Class distribution analysis [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/73ad15e7-45cd-4245-9a22-3ad8135e47de/Lung_Cancer_Model.ipynb)
- Training/validation loss curves [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/73ad15e7-45cd-4245-9a22-3ad8135e47de/Lung_Cancer_Model.ipynb)
- Confusion matrices for all models [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)
- ROC curves and AUC scores
- Precision-Recall curves

## 🔐 Data Privacy

This research adheres to medical data privacy standards:
- No patient identifiable information included
- Dataset used with appropriate permissions
- Compliant with research ethics guidelines

***

**Disclaimer**: This project is for research and educational purposes only. It should not be used for actual clinical diagnosis without supervision from qualified medical professionals. Lung cancer diagnosis requires comprehensive evaluation by healthcare providers. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155601712/375c01a2-217d-4a14-aab8-907a304f0389/conference_latex_template.pdf)

**Version**: 1.0  
**Last Updated**: January 2026  
**Status**: Research Project

***

**Keywords**: Lung Cancer, Deep Learning, SAM, MobileNet, VGG16, DenseNet121, ResNet50, CT Scan Classification, Medical Image Analysis, Computer-Aided Diagnosis
