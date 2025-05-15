# 🔧 Rizz-V: AI-Powered RISC-V Assembly Autocomplete for VS Code

Rizz-V is a Visual Studio Code extension that provides intelligent, context-aware code completions for **RISC-V Assembly language**. It leverages a custom-trained language model served via a local FastAPI backend.

---

## 📦 Features

- 🧠 AI-powered code completion using a local model
- 💡 Suggestions appear when you type a space (`' '`)
- ⚙️ Seamless integration with `.S` and `.asm` files
- 🧪 Easy debugging with VS Code extension development tools

---

## 🧰 Requirements

- Python 3.13
- Node.js & npm
- VS Code

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/GNbros/Rizz-V_AI-Enable
cd Rizz-V_AI-Enable
```

### 2. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
cd rizz-v
npm install
```
### 3. Generate the Model
Open the Jupyter Notebook:
```angular2html
RISC_V_Assembly_Prediction_Model.ipynb
```

Run the cells to either train or load your RISC-V Assembly code model.  
Make sure it generates a working model that can be called via the /generate API later.
### 4. Start the FastAPI Backend
The FastAPI app will expose a /generate endpoint for model inference.  
make sure path is correct in `Rizz-V_AI-Enable`
```bash
uvicorn main:app --reload
```

### 5. Start the VS Code Extension
Open the `rizz-v` directory in VS Code 
```angular2html
cd rizz-v
open .
```
and press `F5` to start the extension in a new window.
This will launch the extension and connect it to the FastAPI backend.
### 6. Test the Extension
Open a `.S` or `.asm` file and start typing.
You should see AI-powered code completions based on the context of your code.
