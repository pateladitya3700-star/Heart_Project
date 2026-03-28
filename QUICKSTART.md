# Quick Start Guide

## Running Locally

### Step 1: Install Python Dependencies
Open your terminal in this project folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: Start the Streamlit App
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Step 3: Use the Application
- Navigate using the top menu buttons
- Upload ECG data files (.zip containing .hea and .dat files)
- View predictions and performance metrics

## Pushing to GitHub

### First Time Setup
```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Create your first commit
git commit -m "Initial commit: ECG heart disease prediction system"

# Create a new repository on GitHub, then link it:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

### Subsequent Updates
```bash
# Stage your changes
git add .

# Commit with a message
git commit -m "Your commit message here"

# Push to GitHub
git push
```

## Notes
- Make sure `Five_Class_Model.h5` exists in the project root for predictions to work
- Large data files in `archive/` are excluded from git (see .gitignore)
- Temporary folders are automatically cleaned by the app
