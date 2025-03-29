# AJL_team_castor-oil Kaggle Project README 

---

### 👥 Team Members

| Name            | Contact                   | Contributions                            |
|-----------------|---------------------------|-------------------------------------------|
| Reva Mahto      | [r-oli-m](https://github.com/r-oli-m) | Model improvement                        |
| Aisha Ahammed   | aahammed@kent.edu         | Data preprocessing, validation strategy   |
| Diego Carillo   | [dicarrillo](https://github.com/dicarrillo) | Exploratory data analysis (EDA)   |
| Nishan Lama     | nishanlama03@gmail.com    | Model improvement, evaluation, GitHub setup |
| Ezuma Ekomo Ble | ezumaekomo01@gmail.com    | Team coordination, fairness analysis      |

---

## 🎯 Project Highlights

- Built a **Convolutional Neural Network (CNN)** using **transfer learning with EfficientNetB0** to classify dermatological images.
- Achieved a **score of 0.51713** and ranked **in the top 25%** of teams on the final Kaggle Leaderboard.
- Applied **Grad-CAM** for visual model explainability.
- Used **label encoding**, **data augmentation**, and **image generators** to preprocess images and improve generalization.

🔗 [AJL 2025 Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

---

## 👩🏽‍💻 Setup & Execution

1. **Clone the repo:**

    ```bash
    git clone https://github.com/nishanlama03/team_castor-oil.git
    cd team_castor-oi
    ```

2. **Install dependencies (Colab recommended):**

    Most dependencies are pre-installed in Colab. If needed:

    ```bash
    pip install kaggle kagglehub tensorflow keras opencv-python
    ```

3. **Access dataset:**

    You must accept the competition rules on Kaggle. Then use:

    ```python
    import kagglehub
    kagglehub.login()
    kagglehub.competition_download('bttai-ajl-2025')
    ```

4. **Run the notebook:**

    Open the provided Colab notebook or `.py` script and run the cells in order.

---

## 🏗️ Project Overview

The AJL Kaggle challenge tasked us with building an equitable dermatology classifier that performs well across all skin tones.

Many dermatology AI models are biased, especially when trained on non-diverse image datasets. This results in unequal diagnostic performance and contributes to systemic health inequities. Our goal was to address this by applying fairness-driven modeling strategies, improving performance for underrepresented skin tones.

---

## 📊 Data Exploration

- Dataset: Provided via Kaggle — 2860 training images, 1227 test images.
- Labels were encoded into numerical form.
- Dataset split: **80% training**, **20% validation**.
- Used **Keras ImageDataGenerators** for real-time image augmentation.
- Sample images were visualized to assess class distribution and quality.

---

## 🧠 Model Development

- Used **EfficientNetB0** for transfer learning with pre-trained ImageNet weights.
- Fine-tuned the top layers for domain-specific features.
- Loss function: `categorical_crossentropy`
- Optimizer: `Adam` with learning rate `1e-4`
- Trained on augmented data (rotation, shift, zoom, etc.)
- Used early stopping and learning rate reduction to avoid overfitting.

---

## 📈 Results & Key Findings

- Achieved **F1 score of 0.51713** on the AJL Kaggle leaderboard.
- **Grad-CAM** helped visualize model focus areas during prediction.
- Evaluated model fairness by inspecting prediction distribution across skin tones.
- Performance improved with added augmented examples for underrepresented groups.

---

### Fairness Interventions:

- Data augmentation for low-represented classes.
- Manual validation of predictions across Fitzpatrick skin types.
- Experimented with class-weighting (less stable).

### Broader Impact:

- Improves trust in AI models in healthcare.
- Encourages equity-centered medical tool design.
- Empowers communities often ignored in mainstream AI development.

---

## 🚀 Next Steps & Future Improvements

- Integrate annotated skin tone labels directly into training.
- Explore ensemble models for misclassification reduction.
- Fine-tune other architectures (ResNet50, ViT).
- Develop fairness tracking metrics within training loop.
- Deploy a demo app to visualize predictions and explanations.

