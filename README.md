# 🥛 Federated Learning Supply Chain 5.0 

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Models-FFD21E?style=for-the-badge&logo=huggingface)
![Unsloth](https://img.shields.io/badge/Unsloth-Fine_Tuning-9CF?style=for-the-badge)

A cutting-edge, AI-driven **Federated Learning Simulation Dashboard** specifically engineered for the Dairy Supply Chain (Milk). 

This project simulates privacy-preserving federated machine learning across three geographically distinct clients (**Amul Gujarat**, **Mother Dairy Delhi**, and **Sudha Bihar**). It actively trains a base Neural Network to optimize profit, reduce waste, and manage carbon caps—while utilizing a **custom Fine-Tuned Qwen2.5-0.5B-Instruct LLM** ("SupplyChainGPT") to act as a brilliant human-in-the-loop strategist over the network's predictions.

---

## 🚀 Key Features

* **🔒 Federated Learning Simulation:** Three active client nodes train a deep neural network privately on their own data. Gradients are aggregated securely using a central server.
* **🛡️ Differential Privacy & Security:** Built-in gradient clipping, Laplace noise injection, and poison-attack thresholds ensure that no single supply client can disrupt or compromise the network.
* **🤖 Custom LLM Agent (SupplyChainGPT):** We stripped standard baseline language models and fine-tuned a custom Qwen2.5-0.5B-Instruct model specifically on heavy milk supply chain data using **Unsloth 4-bit QLoRA**. 
* **⚡ Lightning Fast Offline Inference:** The system loads Merged FP16 `.safetensors` instantly with HuggingFace Transformers directly from the local hard drive for maximum performance.
* **📊 Beautiful Streamlit UI:** Fully interactive dashboard allowing you to watch the federated training live, trigger simulation rounds, and request AI decision overrides.

---

## 🧠 The AI Pipeline

1. **Dataset Generation:** Raw client supply chain CSVs are run through `generate_finetune_dataset.py` to create over 2,000 highly-optimized ChatML instruction-response pairs.
2. **Cloud Training:** `finetune_qwen_supply_chain.py` uses an NVIDIA T4 GPU on Google Colab to attach LoRA adapters and deeply teach the model supply chain dynamics.
3. **Model Centralization:** Unsloth gracefully exports the LoRA weights, unifies them into a **Merged 16-Bit PyTorch Model**, and outputs hyper-fast **GGUF** formats—all neatly centralized into a single folder for seamless deployment.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Prithwiraj731/Federated_Learning_SupplyChain.git
cd Federated_Learning_SupplyChain
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Provide the Fine-Tuned Model Weights
This code relies on the custom AI models that are too large to be stored on GitHub.
1. Make sure you have downloaded the **`SupplyChain_Qwen`** folder.
2. Place the `SupplyChain_Qwen` folder directly into the root directory of this project.

### 5. Launch the Dashboard
```bash
streamlit run app.py
```
*Your browser will cleanly launch the UI. Click "Boot System" to load the local Qwen model into your VRAM.*

---

## 📁 Repository Structure

```text
├── .gitignore                         # Prevents huge weights from bottlenecking Git
├── DATASETS/                          # The raw CSV supply chains for the 3 clients
├── sc50_logs/                         # Auto-generated JSON decision logs from the LLM
├── app.py                             # The primary interactive Streamlit Front-End 
├── main.py                            # The backend configuration and Federated Logic
├── analyze_data.py                    # Legacy data validator
├── generate_datasets.py               # Generates the fake numerical supply data 
├── generate_finetune_dataset.py       # Converts CSV numbers into ChatML for AI reading
└── finetune_qwen_supply_chain.py      # The massive Unsloth Colab SFT Training script
```

---

## 🤝 Acknowledgments
* Massive thanks to **Unsloth AI** for enabling insanely fast 4-bit fine-tuning.
* Built utilizing the **Qwen** foundation architectures.
* User interface beautifully powered by **Streamlit**.
