# 📸 Image Caption Generator

Automatically generate captions for images using deep learning.  
This project combines a **CNN-based feature extractor** and an **LSTM-based language model** to describe images in natural language. The app is built with **Streamlit** for an interactive UI.  

---

## 🚀 Features
- Upload an image (`.jpg`, `.jpeg`, `.png`) and get an **AI-generated caption**.  
- Uses a **CNN encoder** (feature extractor) + **LSTM decoder** (caption generator).  
- Clean, modern **Streamlit UI** with sidebar, styled results, and loading animations.  
- Supports both `.keras` (Keras v3 format) and `.h5` (legacy HDF5 format) model files.  
- Pre-trained tokenizer for converting words ↔ indices.  

---

## 📂 Project Structure
```
.
├── app.py                      # Streamlit app
├── Image_Captioning.ipynb      # Training notebook
├── models/
│   ├── model.keras             # Trained caption model
│   ├── feature_extractor.keras # CNN feature extractor
│   └── tokenizer.pkl           # Tokenizer used for captions
└── README.md                   # Documentation
```

---

## ⚙️ Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt** (example):
```
streamlit
tensorflow
numpy
matplotlib
pillow
pickle5
```

---

## ▶️ Running the App

1. Clone the repository:
   ```bash
   git clone https://github.com/chandan9369/Image-Captioning-App-Using-NLP-and-Computer-Vision.git
   cd Image-Captioning-App-Using-NLP-and-Computer-Vision
   ```

2. Place your trained models inside the `models/` folder:
   - `model.keras` → Caption generation model  
   - `feature_extractor.keras` → CNN feature extractor  
   - `tokenizer.pkl` → Tokenizer used during training  

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at:  
   👉 `http://localhost:8501`

---

## 🧠 Model Workflow
1. **Feature Extraction**  
   - Input image is resized to `(224x224)` and normalized.  
   - A CNN model (e.g., ResNet, VGG, Inception) extracts **feature embeddings**.  

2. **Sequence Generation**  
   - Extracted features + previously generated words are fed into an **LSTM**.  
   - The model predicts the **next word** until it reaches `"endseq"`.  

3. **Tokenizer**  
   - Maps words ↔ integer indices.  
   - Ensures consistent vocabulary between training and inference.  

---

## 🖥️ User Interface
- **Sidebar** → Settings, app info, model description.  
- **Main Area** →  
  - Upload image  
  - Generate caption with **loading spinner**  
  - Display results in **two-column layout** (image + caption card).  

---

## 📘 Example Usage

1. Upload an image of a **dog playing in the park**.  
2. The app may generate:  

   > *"a dog is playing on the grass"* 🐶🌳  

---

## 🛠️ Training (Notebook: `Image_Captioning.ipynb`)
- Preprocess dataset (images + captions).  
- Train CNN feature extractor.  
- Train encoder-decoder (CNN + LSTM).  
- Save:
  - `model.keras`
  - `feature_extractor.keras`
  - `tokenizer.pkl`

---

## 📌 Notes
- If you face issues loading `.keras` models in Streamlit Cloud, use:
  ```python
  load_model("model.keras", compile=False)
  ```
- For older environments, resave your model as `.h5`:
  ```python
  model.save("model.h5")
  ```

---

## 🙌 Acknowledgements
- TensorFlow / Keras team for deep learning frameworks.  
- Streamlit for the interactive web app.  
- Inspiration from various image captioning research papers.  
