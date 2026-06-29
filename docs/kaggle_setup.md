# Kaggle Setup

1. Create a new Kaggle notebook.
2. Enable GPU: Notebook Settings → Accelerator → GPU.
3. Attach datasets:
   - CheXpert-v1.0-small
   - NIH Chest X-rays / ChestX-ray14
4. Clone the GitHub repository:

```bash
git clone https://github.com/<your-username>/em-cl-xray-poc.git
cd em-cl-xray-poc
pip install -r requirements.txt
```

5. Run scripts or notebooks.

Do not store raw medical images in GitHub.
