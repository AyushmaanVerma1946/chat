# 🏥 Medical Chatbot - AI Health Assistant

An **AI-powered medical chatbot** built using **Streamlit** and a **fine-tuned Phi-2 model**. This chatbot provides **basic medical guidance**, answers health-related queries, and ensures **professional tone** while avoiding misinformation.

🚠 **Disclaimer:** This chatbot does *not* replace professional medical advice. Always consult a licensed doctor for health concerns.

---

## 🚀 Features

✅ **Conversational AI** - Ask medical questions, and the chatbot will respond in a structured, professional manner.\
✅ **Medical-Tone Responses** - AI ensures medical accuracy and adds disclaimers where needed.\
✅ **Secure & Responsible AI** - Filters out non-medical or harmful queries (e.g., violence, self-harm).\
✅ **Optimized UI** - Uses Streamlit with **dark theme chat bubbles**, sidebar, and clean interface.\
✅ **Efficient Inference** - Caches model in memory for **fast responses**.

---

## 🖥️ Demo (Screenshots)

| 💬 Chatbot Interface | 🗃️ Sidebar with Info |
| -------------------- | --------------------- |
|                      |                       |

📈 **To add screenshots, save images inside an ****************`assets/`**************** folder.**

---

## 📺 Project Structure

```
📂 chatbot_project
🌍🗃️ data
│   ├── intents.json          # Training dataset
│   ├── tokenized_phi2_data/  # Tokenized dataset (auto-created)
│
🌍🗃️ models
│   ├── phi2_finetuned_model/ # Fine-tuned chatbot model
│
🌍🗃️ scripts
│   ├── train.py              # Fine-tuning script
│   ├── chatbot.py            # Streamlit chatbot UI
│
🌍🗃️ assets                 # Screenshots for README
│
🌍🗃️ logs
│   ├── training_log.txt      # Training logs
│
│── README.md                 # Project documentation
│── requirements.txt          # Python dependencies
```

---

## 🔧 Installation & Setup

### **1️⃣ Install Dependencies**

Make sure you have **Python 3.10+** installed, then run:

```bash
pip install -r requirements.txt
```

### **2️⃣ Run the Chatbot**

```bash
cd scripts
streamlit run chatbot.py
```

This will launch the chatbot in your **web browser**. 🎉

### **3️⃣ Fine-Tune the Model (Optional)**

If you want to fine-tune the chatbot on **your own medical dataset**, run:

```bash
python scripts/train.py
```

---

## 🎨 Customization

### **Change Chat Colors**

Edit `chatbot.py` to modify chat bubble colors:

```python
background-color: #006400;  /* Dark Green for AI */
background-color: #333333;  /* Dark Gray for User */
```

Change the hex codes to **customize the chat UI**.

---

## 🗃️ License

This project is **open-source** under the **MIT License**.

---

## 👨‍💻 Contributors

- **[Your Name]** - Developer & AI Engineer
- **[Your Team Members (if any)]**

---

## ❓ FAQ

### **Q1: Can I use this chatbot for real medical consultations?**

🚫 **No!** This chatbot provides **basic guidance only**. Always seek professional medical advice.

### **Q2: Can I train the chatbot on my own dataset?**

👌 **Yes!** Modify `intents.json` and run `train.py` to fine-tune it.

### **Q3: How can I deploy this chatbot online?**

🚀 Deploy it on **Hugging Face Spaces** or **Streamlit Cloud** for public use.

---

📩 **Need Help?** Contact me at vayushmaan.verma57@gmail.com
# Chat-with-me
