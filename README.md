# Iris Classifier

Simple Streamlit data science project.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py


iris-classifier-streamlit/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── Procfile                # For Render
├── runtime.txt             # Python version for Render
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions pipeline
├── data/
│   └── iris.csv            # (optional, download it)
├── train.py                # Optional: separate training script

