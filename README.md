# 🌿 Plant Disease Detection — End-to-End Deep Learning Project

This project presents a deep learning approach for detecting plant diseases from leaf images using a Convolutional Neural Network (CNN).

The model classifies plant leaf images into disease categories and predicts the corresponding class with a confidence score.


## 🚀 Project Highlights

- Real-world agricultural image classification problem
- Custom CNN model developed using TensorFlow/Keras
- Dataset containing 6,500 leaf images across 5 classes
- Image preprocessing, normalization, batching, and augmentation
- Achieved 92%+ validation accuracy
- End-to-end workflow covering preprocessing, training, evaluation, and inference
  

## 🧠 Deep Learning Workflow

- Dataset loading and exploration
- Image preprocessing and normalization
- Training-validation split (80:20)
- CNN architecture development
- Model training and evaluation
- Accuracy and loss visualization
- Prediction pipeline for unseen images
- Model serialization for future deployment


## 🤖 Model Used

Convolutional Neural Network (CNN) :

- Convolution Layers
- Max Pooling Layers
- Dense Layers
- Dropout Regularization
- Softmax Output Layer

CNN was selected because it efficiently learns spatial patterns and performs well in image classification tasks.


## 📥 Input

Plant leaf image


## 📊 Dataset Information

| Metric            | Value     |
| ----------------- | --------- |
| Total Images      | 6500      |
| Classes           | 5         |
| Training Images   | 5200      |
| Validation Images | 1300      |
| Image Size        | 224 × 224 |


## 📈 Output Provided

The model predicts:

- Disease name (class label)
- Confidence score (prediction probability)

Example output:

- Prediction: Tomato Leaf Blight  
- Confidence: 91.8%


## 📂 Project Structure

Plant-Disease-Detection/
│
├── dataset/
│   └── sample_images/
│       ├── potato_late_blight.jfif
│       ├── tomato_early_blight.jfif
│       └── tomato_healthy.jfif
│
├── notebook/
│   └── Plant_Disease_Detection.ipynb
│
├── outputs/
│   ├── accuracy_curve.png
│   └── sample_predictions.png
│
├── screenshots/
│   ├── dataset structure.PNG
│   ├── prediction output.PNG
│   └── training result.PNG
│
├── README.md
├── requirements.txt
└── .gitignore

## ⚙️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook


## 🎯 Business Value

Plant disease detection helps:

- Farmers identify diseases early
- Reduce crop loss
- Improve agricultural productivity
- Enable smart farming solutions


## 👨‍💻 Author

Deep Learning project developed to explore practical applications of computer vision in agriculture.
