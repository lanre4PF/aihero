# 🤖 AI FAQ Assistant

A Streamlit chatbot that answers questions about the [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) repository using AI-powered search.

## Features

- Interactive chat interface (Streamlit)
- Answers questions about the Ultralytics repo
- Caches and indexes repo docs for fast search

## Quickstart

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/app
```

### 2. Install dependencies

Make sure you have Python 3.8+ and [pip](https://pip.pypa.io/en/stable/).

```bash
pip install -r ../requirements.txt
```
Or manually install:
```bash
pip install streamlit frontmatter requests minsearch
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at [http://localhost:8501](http://localhost:8501).

## Usage

- Type your question in the chat box.
- The assistant will answer using information from the Ultralytics repository.

## Notes

- The first run may take a while as it downloads and indexes the repo.
- For cloud deployment, just upload this repo to [Streamlit Community Cloud](https://streamlit.io/cloud) and click "Deploy".

---

**Made with ❤️ using Streamlit and AI search.**