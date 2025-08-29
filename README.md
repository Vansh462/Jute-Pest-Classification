# 🐛 Jute Pest Classification System

An intelligent web application for identifying and classifying 17 different types of jute pests using a state-of-the-art BiT (Big Transfer) deep learning model. This system provides farmers and agricultural experts with an easy-to-use tool for rapid pest identification.

## ✨ Features

- **🎯 High Accuracy**: 95.5% test accuracy with BiT-M R101x1 model
- **⚡ Real-time Classification**: Instant pest identification from uploaded images
- **📊 Confidence Scoring**: Detailed probability scores for all predictions
- **🏆 Top-3 Predictions**: Alternative possibilities with confidence levels
- **📱 User-friendly Interface**: Clean, intuitive web interface
- **🔄 Smart Caching**: Fast subsequent predictions after initial model load
- **📈 Detailed Analytics**: Complete probability breakdown for all 17 pest classes

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (for model loading)
- Modern web browser

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Jute-Pest-Classification
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

4. **Access the app**:
   - Open your browser and navigate to `http://localhost:8501`
   - The model will take 10-15 seconds to load initially
   - Subsequent runs will be much faster due to caching

## 📁 Project Structure

```
Jute-Pest-Classification/
├── 📄 app.py                     # Main Streamlit application
├── 📓 JP.ipynb                   # Jupyter notebook for model training
├── 📋 requirements.txt           # Python dependencies
├── 🗂️ Jute_Pest_Dataset/         # Training dataset
│   ├── train/                   # Training images
│   ├── val/                     # Validation images
│   └── test/                    # Test images
├── 🤖 my_saved_bit_model/        # Trained BiT model (~170MB)
├── 📊 logs/                      # Training logs and metrics
├── ⚡ convert_to_tflite.py       # Model optimization script
├── 🔧 optimize_model.py          # Model compression utility
└── 📖 README.md                  # This documentation
```

## 🐛 Supported Pest Classifications

The system can accurately identify and classify **17 different jute pest species**:

| # | Pest Name | Category | Damage Type |
|---|-----------|----------|-------------|
| 1 | **Beet Armyworm** | Lepidoptera | Leaf feeder |
| 2 | **Black Hairy** | Caterpillar | Defoliator |
| 3 | **Cutworm** | Lepidoptera | Root/stem cutter |
| 4 | **Field Cricket** | Orthoptera | Omnivorous |
| 5 | **Jute Aphid** | Hemiptera | Sap sucker |
| 6 | **Jute Hairy** | Caterpillar | Leaf feeder |
| 7 | **Jute Red Mite** | Acari | Sap sucker |
| 8 | **Jute Semilooper** | Lepidoptera | Defoliator |
| 9 | **Jute Stem Girdler** | Coleoptera | Stem borer |
| 10 | **Jute Stem Weevil** | Coleoptera | Stem borer |
| 11 | **Leaf Beetle** | Coleoptera | Leaf feeder |
| 12 | **Mealybug** | Hemiptera | Sap sucker |
| 13 | **Pod Borer** | Lepidoptera | Pod/seed feeder |
| 14 | **Scopula Emissaria** | Lepidoptera | Defoliator |
| 15 | **Termite** | Isoptera | Wood/cellulose feeder |
| 16 | **Termite odontotermes (Rambur)** | Isoptera | Soil/root feeder |
| 17 | **Yellow Mite** | Acari | Sap sucker |

## 📊 Model Performance & Specifications

### Performance Metrics
- **🎯 Test Accuracy**: 95.5%
- **⚡ Inference Time**: 1-3 seconds per image
- **🧠 Model Architecture**: BiT-M R101x1 (Big Transfer)
- **📐 Input Resolution**: 512×512 pixels
- **💾 Model Size**: ~170MB
- **🔄 Loading Time**: 10-15 seconds (first run), instant (cached)

### Technical Details
- **Framework**: TensorFlow 2.13+
- **Pre-training**: ImageNet-21k
- **Fine-tuning**: Custom jute pest dataset
- **Optimization**: Mixed precision training
- **Data Augmentation**: Advanced augmentation pipeline

## 🎯 How to Use

### Step-by-Step Guide

1. **🚀 Launch the Application**
   ```bash
   streamlit run app.py
   ```

2. **⏳ Wait for Model Loading**
   - First run: 10-15 seconds (model loads from disk)
   - Subsequent runs: Instant (cached in memory)

3. **📤 Upload an Image**
   - Click "Browse files" or drag & drop
   - Supported formats: JPG, JPEG, PNG, BMP
   - Recommended: Clear, well-lit images

4. **🔍 View Results**
   - **Primary Prediction**: Most likely pest type
   - **Confidence Score**: Model certainty (0-100%)
   - **Top 3 Predictions**: Alternative possibilities
   - **Full Analysis**: Complete probability breakdown

### 📸 Image Guidelines

| ✅ **Good Images** | ❌ **Avoid** |
|-------------------|-------------|
| Clear, focused pest | Blurry or out-of-focus |
| Good lighting | Too dark/bright |
| Pest fills frame | Pest too small |
| Single pest visible | Multiple pests |
| Natural colors | Heavy filters |

## ⚡ Performance Optimization

### Model Size Optimization

For faster loading times, you can create optimized model versions:

1. **TensorFlow Lite Conversion** (Recommended):
   ```bash
   python convert_to_tflite.py
   ```
   - Reduces model size by 50-80%
   - Faster loading and inference
   - Maintains high accuracy

2. **Model Compression**:
   ```bash
   python optimize_model.py
   ```
   - Creates optimized SavedModel
   - Better memory efficiency

## 🔧 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Model loading fails** | Ensure `my_saved_bit_model/` contains all files |
| **Slow first load** | Normal behavior - subsequent loads are cached |
| **Low accuracy** | Use clear, well-lit images with visible pests |
| **Memory errors** | Ensure 4GB+ RAM available |
| **Import errors** | Run `pip install -r requirements.txt` |

### Model File Structure
Your `my_saved_bit_model/` should contain:
```
my_saved_bit_model/
├── saved_model.pb          # Model architecture
├── fingerprint.pb          # Model fingerprint
└── variables/              # Model weights
    ├── variables.index
    └── variables.data-00000-of-00001
```

### Re-training the Model

If you need to retrain or modify the model:

1. **Open the training notebook**:
   ```bash
   jupyter notebook JP.ipynb
   ```

2. **Save model after training**:
   ```python
   # In your notebook after training
   tf.saved_model.save(model, 'my_saved_bit_model')
   ```

### API Integration

The core prediction function can be used independently:

```python
from app import load_model, preprocess_image, predict_pest

# Load model once
model = load_model()

# Predict on new image
image_array = preprocess_image(your_image)
pred_idx, confidence, probabilities = predict_pest(model, image_array)
```

## 📊 Dataset Information

- **Total Images**: Varies by class
- **Image Resolution**: 512×512 pixels
- **Format**: RGB color images
- **Split**: Train/Validation/Test
- **Augmentation**: Rotation, flip, zoom, brightness

## 🔬 Research & Citations

This project uses the BiT (Big Transfer) model architecture:

```bibtex
@article{kolesnikov2020big,
  title={Big Transfer (BiT): General Visual Representation Learning},
  author={Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Puigcerver, Joan and Yung, Jessica and Gelly, Sylvain and Houlsby, Neil},
  journal={ECCV},
  year={2020}
}
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **🍴 Fork the repository**
2. **🌿 Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **💻 Make your changes**
4. **✅ Test thoroughly**
5. **📝 Commit your changes**:
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **🚀 Push to the branch**:
   ```bash
   git push origin feature/amazing-feature
   ```
7. **🔄 Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/Jute-Pest-Classification.git
cd Jute-Pest-Classification

# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Start development server
streamlit run app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** for the excellent deep learning framework
- **Streamlit Team** for the amazing web app framework
- **Google Research** for the BiT model architecture
- **Agricultural Research Community** for pest classification datasets

## 📞 Support & Contact

- **🐛 Bug Reports**: [Open an issue](https://github.com/yourusername/Jute-Pest-Classification/issues)
- **💡 Feature Requests**: [Start a discussion](https://github.com/yourusername/Jute-Pest-Classification/discussions)
- **📧 Email**: your.email@example.com

---

<div align="center">

**⭐ Star this repository if it helped you!**

Made with ❤️ for the agricultural community

</div>