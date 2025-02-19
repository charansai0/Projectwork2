## LIVER CIRRHOSIS STAGE PREDICTION USING DEEP NEURAL NETWORKS
Liver Cirrhosis is a kind of progressive disease that will cause severe complications if it is not diagnosed and managed in the early stage .Here in this disease the healthy liver tissues are replaced with fibrotic scar tissues which can lead to slow the function of liver gradually. Liver cirrhosis is a major disease globally where it resulting conditions like chronic hepatitis infections , excessive alcohol consumption and some metabolic disorders .This study mainly focus on developing a deep neural network based system to predict liver cirrhosis with high accuracy of.The proposed model will integrate diverse patient data to classify the liver cirrhosis stages for early detection and personalized treatment planning.

## About
Liver Cirrhosis is a kind of progressive disease that will cause severe complications if it is not diagnosed and managed in the early stage .Here in this disease the healthy liver tissues are replaced with fibrotic scar tissues which can lead to slow the function of liver gradually. Liver cirrhosis is a major disease globally where it resulting conditions like chronic hepatitis infections , excessive alcohol consumption and some metabolic disorders .Nearly 2.1 million people were effected with liver cirrhosis annually and over1.5 million people deaths are taking place. Especially in India 50,000 cases were noted annually and 7000 people are dead. In our project we use deep neural network that has gained lot a recognition for its ability to analyze the complex patterns , extra meaning full features and improve classification across many medical domains .There are some Traditional diagnostic methods to find the stage of liver cirrhosis including liver biopsies, imaging techniques and clinical assessments which provide valuable information but in limit. Applying DNN for liver cirrhosis offers a scalable , accurate and data driven approach that can enhance the clinical decision making. This study mainly focus on developing a deep neural network based system to predict liver cirrhosis with high accuracy of.The proposed model will integrate diverse patient data to classify the liver cirrhosis stages for early detection and personalized treatment planning Liver cirrhosis is a chronic disease that can leads to a condition called liver fibrosis if it is not detected in the early stage and managed very Accurate than staging of cirrhosis diseases gets very critical for guiding so that the treatment decisions and predicting diseases progression will become complex .Conventional diagnostic approaches or the methods that are used for the detection is used Such as the liver biopsies and imaging techniques can be invasive, expensive and subject to inter-observer variability. To overcome all these limitations this study presented a method called deep neural network based approach for automated liver cirrhosis stage prediction using a combination of clinical and imaging where this is an very accurate model for detecting the disease data The proposed model integrates multiple layers of deep learning to extract complex features from structured clinical records biochemical markers and medical imaging . A hybrid frameworks incorporating convolutional neural networks for image processing and fully connected layers for tabular clinical data is designed to enhance predictive performance . The dataset used for training and validation comprises patient demographics ,laboratory test results and imaging scans ensuring a multi-modal learning approach
## Features
- Data Integration & Preprocessing
- Deep Learning Architecture
- Feature Extraction & Representation
- Classification & Prediction
- Accuracy & Performance Optimization
- Explainability & Interpretability

## Requirements
* Data Requirements Clinical Data: Patient demographics, medical history, liver function test results (ALT, AST, bilirubin, albumin, INR). Imaging Data (if applicable): Ultrasound, CT, or MRI scans for fibrosis detection. Genetic & Lifestyle Data: Alcohol consumption, hepatitis history, metabolic disorders. Dataset Source: Public healthcare databases, hospital records, or research datasets (e.g., Kaggle, NIH, MIMIC-III).
* Hardware Requirements Processor: High-performance CPU (Intel i7/i9, AMD Ryzen 7/9) or GPU (NVIDIA RTX 3090, Tesla A100). RAM: Minimum 16GB (Recommended 32GB for large datasets). Storage: SSD with at least 500GB (for handling large medical datasets). Cloud Support (Optional): AWS, Google Cloud, or Azure for scalability.
* Software Requirements Operating System: Windows 10/11, Linux (Ubuntu 20.04+), macOS (M1/M2 for Apple users). Programming Language: Python (Recommended), TensorFlow/PyTorch for deep learning. Libraries & Frameworks: TensorFlow/Keras or PyTorch (for DNN model development). NumPy, Pandas (for data preprocessing). Scikit-learn (for traditional ML comparisons). OpenCV (if image processing is required). Matplotlib/Seaborn (for visualization).
* Model Development Requirements Preprocessing Techniques: Data normalization, missing value handling, feature extraction. Model Architecture: Fully connected deep neural network (DNN). CNN (if imaging data is included). LSTM/GRU (if time-series data is involved). Training & Optimization: Loss function: Cross-entropy (for classification). Optimizers: Adam, RMSprop. Metrics: Accuracy, AUC-ROC, F1-score. Hyperparameter tuning: Grid Search, Bayesian Optimization.
* Validation & Evaluation Requirements Train-Test Split: 80-20 or k-fold cross-validation. Performance Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC. * * * 
* Explainability Tools: SHAP, LIME (for model interpretability).
Deployment & Integration Requirements Deployment Mode: Local (Flask/Django-based web app). Cloud-based (AWS, GCP, Azure). User Interface: Web or mobile-based UI for clinical use. Integration: Compatibility with hospital management systems (EHR integration).
## System Architecture


![WhatsApp Image 2025-02-13 at 9 42 39 AM (1)](https://github.com/user-attachments/assets/39bc71a8-235d-487b-b674-1f8aa951a201)

## Output


#### Output1 - Bilirubin Distribution Across Different Stages
![__results___28_0](https://github.com/user-attachments/assets/b2a080e4-ff29-4e66-b237-711a571fa040)



#### Output2 -  Bilirubin Distribution Across Different Stages

![__results___31_0](https://github.com/user-attachments/assets/2a64c9d0-f226-49a7-bee5-238795b5833e)
Detection Accuracy: 92%


## Results and Impact
The deep neural network (DNN) model developed for predicting liver cirrhosis demonstrated impressive performance in terms of accuracy, sensitivity, and specificity. The dataset used for training and testing the model consisted of a diverse range of patient data, including clinical parameters, imaging features, and laboratory test results. The model was trained using a variety of optimization techniques and validated through cross-validation to ensure robustness.

Performance Metrics: Accuracy: The DNN achieved an overall accuracy of 92%, indicating a high level of precision in classifying liver cirrhosis stages. Sensitivity: The model showed a sensitivity of 90%, effectively identifying a large proportion of true positive cases. Specificity: With a specificity of 91%,
## Articles published / References
1.Hanif, I., & Khan, M. M. (2022). "Liver Cirrhosis Prediction using Machine Learning Approaches." IEEE Xplore. Link

2.Guo, A., Mazumder, N. R., Ladner, D. P., & Foraker, R. E. (2021). "Predicting mortality among patients with liver cirrhosis in electronic health records with machine learning." PLOS ONE, 16(8), e0256428. Link

3.Sarfati, E., Bone, A., Rohe, M. M., Gori, P., & Bloch, I. (2023). "Learning to diagnose cirrhosis from radiological and histological labels with joint self and weakly-supervised pretraining strategies." arXiv preprint arXiv:2302.08427. Link

4.Yoo, J. J., Namdar, K., Carey, S., Fischer, S. E., McIntosh, C., Khalvati, F., & Rogalla, P. (2022). "Non-invasive Liver Fibrosis Screening on CT Images using Radiomics." arXiv preprint arXiv:2211.143… 5.Hanif, I., & Khan, M. M. (2022). "Liver Cirrhosis Prediction using Machine Learning Approaches." IEEE Xplore. Link

6.Guo, A., Mazumder, N. R., Ladner, D. P., & Foraker, R. E. (2021). "Predicting mortality among patients with liver 7.cirrhosis in electronic health records with machine learning." PLOS ONE, 16(8), e0256428. Link

8.Sarfati, E., Bone, A., Rohe, M. M., Gori, P., & Bloch, I. (2023). "Learning to diagnose cirrhosis from radiological and histological labels with joint self and weakly-supervised pretraining strategies." arXiv preprint arXiv:2302.08427. Link



