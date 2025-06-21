# 🎯 ThreatTextAI

**ThreatTextAI** is a powerful Python project that uses a BERT model to classify text messages by threat types: **attacks**, **movements**, **disinformation**, and **neutral texts**. 🚀

## 🌟 Description
This project implements machine learning for text analysis using the `train.csv` dataset and provides a convenient tool for:
- 🏋️‍♂️ Training the model on training data.
- 🔮 Working with new texts and classifying them.
- ✅ Evaluating the model on test data from the `test.csv` file.

## 🛠️ Requirements
- **Python 3.9+ (up to 11.9)**
- Libraries:
  - `torch` (version 2.5.0 or compatible)
  - `transformers`
  - `pandas`
  - `scikit-learn`
  - `nlpaug` (for data augmentation, optional)

## 📦 Installation
1. Clone the repository:

`git clone https://github.com/Leeva13/ThreatTextAI.git`

`cd ThreatTextAI`

2. Create and activate a virtual environment:

`python -m venv .venv`

`.venv\Scripts\activate`  # On Windows

3. Install dependencies:

`pip install -r requirements.txt`

4. Prepare the data:
- The file `data/train.csv` with columns `text` and `label` (0=attack, 1=movement, 2=disinfo, 3=neutral).
- The file `data/test.csv` (optional) with the same columns for testing.

### 📋 Data Example
| `text`                                      | `label`  |
|---------------------------------------------|----------|
| "Shelling of Kharkiv with MLRS – infrastructure damaged" | `attack` (0) |
| "A column of Russian tanks spotted on the highway near Svatove"  | `movement` (1) |
| "Ukraine destroyed its own infrastructure to blame Russia" | `disinfo` (2) |
| "It's sunny today, a good day for a walk"      | `neutral` (3) |

## ▶️ Running
1. Activate the virtual environment:

`.venv\Scripts\activate`

2. Run the program:

`python src/main.py`

3. Choose an option in the menu:
- `1️⃣` Train the model on `train.csv`.
- `2️⃣` Enter text for classification.
- `3️⃣` Evaluate the model on `test.csv`.
- `4️⃣` Exit.

## 📂 Project Structure
- `src/main.py`: Main script with menu.  
- `src/train.py`: Model training logic.  
- `src/predict.py`: Prediction function (optional).  
- `src/dataset.py`: Data processing class.  
- `data/train.csv`: Training data.  
- `data/test.csv`: Test data (optional).  
- `models/`: Folder for saving the trained model (`best_model.pth`).

## 💡 Improvements
- ➕ Add more data to `train.csv` and `test.csv` for better model generalization.
- ⚡ Try setting up GPU (NVIDIA CUDA) to speed up training.

## 👤 Author
- **Leeva13**  
- Contact: [artembrk11@gmail.com](mailto:artembrk11@gmail.com) or [GitHub](https://github.com/Leeva13)