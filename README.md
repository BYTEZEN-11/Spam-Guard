# Spam Email Classifier

Machine learning system for email classification using multiple algorithms (SVM, Logistic Regression, Decision Tree, KNN, Random Forest) with an interactive Streamlit interface.

## Features

- Real-time email classification with confidence scores
- Batch processing for MBOX files
- Automated model training with hyperparameter tuning
- Performance metrics logging (Accuracy, Precision, Recall, F1-Score)

## Tech Stack

Python 3.10+ | Streamlit | Scikit-learn | Pandas | NumPy | BeautifulSoup4

## Quick Start

```bash
git clone <repository_url>
cd Spam-Email-Detection
pip install -r requirements.txt
streamlit run app.py
```

## Usage

**Classify Single Email**: Paste content and get instant prediction
**Batch Processing**: Upload MBOX file for multiple email analysis

## Training

```bash
python -m src.pipeline.training_pipeline
```

Models are saved to `outputs/` with evaluation metrics.

## Configuration

Edit `src/config/config.py` to adjust:
- Model hyperparameters
- File paths
- Cross-validation settings

## License

MIT License

<!-- Build step 23: Create model training component skeleton -->

<!-- Build step 24: Implement Logistic Regression model training routine -->

<!-- Build step 25: Add Naive Bayes classifier training logic -->

<!-- Build step 26: Support Support Vector Machine classifier training -->

<!-- Build step 27: Implement Random Forest classifier training routine -->

<!-- Build step 28: Add K-Nearest Neighbors classifier support -->

<!-- Build step 29: Implement Decision Tree classifier training function -->

<!-- Build step 30: Add evaluation metrics calculation for trained models -->

<!-- Build step 31: Include confusion matrix export in model evaluation -->

<!-- Build step 32: Add classification report generator helper -->

<!-- Build step 33: Create prediction pipeline initialization module -->

<!-- Build step 34: Implement single email prediction interface -->

<!-- Build step 35: Add batch email prediction handler for dataframe input -->

<!-- Build step 36: Support MBOX file format parsing in prediction pipeline -->

<!-- Build step 37: Implement stream reading for large MBOX files -->

<!-- Build step 38: Add email header extraction routines for MBOX parser -->

<!-- Build step 39: Create Streamlit web application template -->

<!-- Build step 40: Add single email classification UI tab in app -->

<!-- Build step 41: Add batch MBOX file upload widget in Streamlit UI -->

<!-- Build step 42: Implement confidence score display in dashboard -->

<!-- Build step 43: Add warning modal for oversized file uploads -->

<!-- Build step 44: Set default page configuration and branding for app -->

<!-- Build step 45: Implement cached pipeline loader for faster app startup -->

<!-- Build step 46: Add error handling for unreadable MBOX attachments -->

<!-- Build step 47: Include progress bar during batch email prediction -->

<!-- Build step 48: Add export option for predictions to CSV file -->

<!-- Build step 49: Enhance logger formatting with timestamps and levels -->

<!-- Build step 50: Add file path validator in configuration script -->

<!-- Build step 51: Improve error messaging for invalid text input -->

<!-- Build step 52: Refactor TF-IDF vectorization parameter defaults -->

<!-- Build step 53: Update model hyperparameter tuning grid search -->

<!-- Build step 54: Add cross-validation fold configuration options -->

<!-- Build step 55: Create pipeline output directory structure automated creator -->

<!-- Build step 56: Optimize MBOX temporary file cleanup logic -->

<!-- Build step 57: Add sample dataset CSV file to repository -->

<!-- Build step 58: Include pre-trained vectorizer artifact in data models -->

<!-- Build step 59: Add trained model pickle artifact for version 1 -->

<!-- Build step 60: Create log storage directory with sample log entry -->

<!-- Build step 61: Update requirements file with pinned dependency versions -->

<!-- Build step 62: Enhance README with usage instructions for batch processing -->

<!-- Build step 63: Add section on model re-training to README -->

<!-- Build step 64: Update pyproject metadata with author and version -->

<!-- Build step 65: Refactor logger import statement in app module -->

<!-- Build step 66: Add input length validation in Streamlit text input area -->

<!-- Build step 67: Improve spam vs ham label formatting in UI output -->

<!-- Build step 68: Add metric visualization charts in training pipeline -->

<!-- Build step 69: Implement model artifact serialization checks -->

<!-- Build step 70: Support custom model artifact loading path setting -->

<!-- Build step 71: Fix potential memory leak during MBOX batch iteration -->

<!-- Build step 72: Clean up unused helper functions in utils script -->

<!-- Build step 73: Add inline documentation docstrings across components -->

<!-- Build step 74: Standardize exception logging format across pipelines -->

<!-- Build step 75: Implement environment variable override for paths -->

<!-- Build step 76: Refactor dataset loading exception handling -->

<!-- Build step 77: Optimize regex patterns for email link extraction -->

<!-- Build step 78: Add unit test suite setup for preprocessor module -->

<!-- Build step 79: Implement feature extraction validation checks -->

<!-- Build step 80: Enhance Streamlit theme styling and icon display -->

<!-- Build step 81: Add dataset balance statistics printer in ingestion -->

<!-- Build step 82: Support configurable max features in vectorizer -->

<!-- Build step 83: Implement graceful fallback when model files missing -->

<!-- Build step 84: Optimize memory footprint during large text vectorization -->

<!-- Build step 85: Refactor prediction pipeline output dictionary schema -->

<!-- Build step 86: Add retry logic for file file-system operations -->

<!-- Build step 87: Update gitignore to exclude local log outputs -->

<!-- Build step 88: Enhance log level configuration support via config -->

<!-- Build step 89: Add parameter checks for model evaluation function -->

<!-- Build step 90: Standardize return types in email processing helpers -->

<!-- Build step 91: Optimize batch prediction speed using vectorized operations -->

<!-- Build step 92: Clean up temporary directory references post batch run -->

<!-- Build step 93: Add comprehensive comment header to entrypoint app -->

<!-- Build step 94: Refactor model loading caching decorator mechanism -->

<!-- Build step 95: Update model version folder naming convention -->

<!-- Build step 96: Implement input sanitization before vectorization step -->

<!-- Build step 97: Enhance error handling for empty text payloads -->

<!-- Build step 98: Finalize README quick start and architecture overview -->

<!-- Build step 99: Perform final codebase cleanup and code formatting -->

<!-- Build step 100: Finalize initial release build for Spam Guard system -->

<!-- Build step 23: Create model training component skeleton -->

<!-- Build step 24: Implement Logistic Regression model training routine -->

<!-- Build step 25: Add Naive Bayes classifier training logic -->

<!-- Build step 26: Support Support Vector Machine classifier training -->

<!-- Build step 27: Implement Random Forest classifier training routine -->

<!-- Build step 28: Add K-Nearest Neighbors classifier support -->

<!-- Build step 29: Implement Decision Tree classifier training function -->

<!-- Build step 30: Add evaluation metrics calculation for trained models -->

<!-- Build step 31: Include confusion matrix export in model evaluation -->

<!-- Build step 32: Add classification report generator helper -->

<!-- Build step 33: Create prediction pipeline initialization module -->

<!-- Build step 34: Implement single email prediction interface -->

<!-- Build step 35: Add batch email prediction handler for dataframe input -->

<!-- Build step 36: Support MBOX file format parsing in prediction pipeline -->

<!-- Build step 37: Implement stream reading for large MBOX files -->

<!-- Build step 38: Add email header extraction routines for MBOX parser -->

<!-- Build step 39: Create Streamlit web application template -->

<!-- Build step 40: Add single email classification UI tab in app -->

<!-- Build step 41: Add batch MBOX file upload widget in Streamlit UI -->

<!-- Build step 42: Implement confidence score display in dashboard -->

<!-- Build step 43: Add warning modal for oversized file uploads -->

<!-- Build step 44: Set default page configuration and branding for app -->

<!-- Build step 45: Implement cached pipeline loader for faster app startup -->

<!-- Build step 46: Add error handling for unreadable MBOX attachments -->

<!-- Build step 47: Include progress bar during batch email prediction -->

<!-- Build step 48: Add export option for predictions to CSV file -->

<!-- Build step 49: Enhance logger formatting with timestamps and levels -->

<!-- Build step 50: Add file path validator in configuration script -->

<!-- Build step 51: Improve error messaging for invalid text input -->

<!-- Build step 52: Refactor TF-IDF vectorization parameter defaults -->

<!-- Build step 53: Update model hyperparameter tuning grid search -->

<!-- Build step 54: Add cross-validation fold configuration options -->

<!-- Build step 55: Create pipeline output directory structure automated creator -->

<!-- Build step 56: Optimize MBOX temporary file cleanup logic -->

<!-- Build step 57: Add sample dataset CSV file to repository -->

<!-- Build step 58: Include pre-trained vectorizer artifact in data models -->

<!-- Build step 59: Add trained model pickle artifact for version 1 -->

<!-- Build step 60: Create log storage directory with sample log entry -->

<!-- Build step 61: Update requirements file with pinned dependency versions -->

<!-- Build step 62: Enhance README with usage instructions for batch processing -->

<!-- Build step 63: Add section on model re-training to README -->

<!-- Build step 64: Update pyproject metadata with author and version -->

<!-- Build step 65: Refactor logger import statement in app module -->

<!-- Build step 66: Add input length validation in Streamlit text input area -->

<!-- Build step 67: Improve spam vs ham label formatting in UI output -->

<!-- Build step 68: Add metric visualization charts in training pipeline -->

<!-- Build step 69: Implement model artifact serialization checks -->

<!-- Build step 70: Support custom model artifact loading path setting -->

<!-- Build step 71: Fix potential memory leak during MBOX batch iteration -->

<!-- Build step 72: Clean up unused helper functions in utils script -->

<!-- Build step 73: Add inline documentation docstrings across components -->

<!-- Build step 74: Standardize exception logging format across pipelines -->

<!-- Build step 75: Implement environment variable override for paths -->

<!-- Build step 76: Refactor dataset loading exception handling -->

<!-- Build step 77: Optimize regex patterns for email link extraction -->

<!-- Build step 78: Add unit test suite setup for preprocessor module -->

<!-- Build step 79: Implement feature extraction validation checks -->

<!-- Build step 80: Enhance Streamlit theme styling and icon display -->

<!-- Build step 81: Add dataset balance statistics printer in ingestion -->

<!-- Build step 82: Support configurable max features in vectorizer -->

<!-- Build step 83: Implement graceful fallback when model files missing -->

<!-- Build step 84: Optimize memory footprint during large text vectorization -->

<!-- Build step 85: Refactor prediction pipeline output dictionary schema -->

<!-- Build step 86: Add retry logic for file file-system operations -->

<!-- Build step 87: Update gitignore to exclude local log outputs -->

<!-- Build step 88: Enhance log level configuration support via config -->

<!-- Build step 89: Add parameter checks for model evaluation function -->

<!-- Build step 90: Standardize return types in email processing helpers -->

<!-- Build step 91: Optimize batch prediction speed using vectorized operations -->

<!-- Build step 92: Clean up temporary directory references post batch run -->

<!-- Build step 93: Add comprehensive comment header to entrypoint app -->

<!-- Build step 94: Refactor model loading caching decorator mechanism -->

<!-- Build step 95: Update model version folder naming convention -->

<!-- Build step 96: Implement input sanitization before vectorization step -->

<!-- Build step 97: Enhance error handling for empty text payloads -->

<!-- Build step 98: Finalize README quick start and architecture overview -->

<!-- Build step 99: Perform final codebase cleanup and code formatting -->

<!-- Build step 100: Finalize initial release build for Spam Guard system -->

<!-- Build step 22: Add tokenization utility functions for raw body text -->

<!-- Build step 23: Create model training component skeleton -->

<!-- Build step 24: Implement Logistic Regression model training routine -->

<!-- Build step 25: Add Naive Bayes classifier training logic -->

<!-- Build step 26: Support Support Vector Machine classifier training -->

<!-- Build step 27: Implement Random Forest classifier training routine -->

<!-- Build step 28: Add K-Nearest Neighbors classifier support -->

<!-- Build step 29: Implement Decision Tree classifier training function -->

<!-- Build step 30: Add evaluation metrics calculation for trained models -->

<!-- Build step 31: Include confusion matrix export in model evaluation -->

<!-- Build step 32: Add classification report generator helper -->

<!-- Build step 33: Create prediction pipeline initialization module -->

<!-- Build step 34: Implement single email prediction interface -->

<!-- Build step 35: Add batch email prediction handler for dataframe input -->

<!-- Build step 36: Support MBOX file format parsing in prediction pipeline -->

<!-- Build step 37: Implement stream reading for large MBOX files -->

<!-- Build step 38: Add email header extraction routines for MBOX parser -->

<!-- Build step 39: Create Streamlit web application template -->

<!-- Build step 40: Add single email classification UI tab in app -->

<!-- Build step 41: Add batch MBOX file upload widget in Streamlit UI -->

<!-- Build step 42: Implement confidence score display in dashboard -->

<!-- Build step 43: Add warning modal for oversized file uploads -->

<!-- Build step 44: Set default page configuration and branding for app -->

<!-- Build step 45: Implement cached pipeline loader for faster app startup -->

<!-- Build step 46: Add error handling for unreadable MBOX attachments -->

<!-- Build step 47: Include progress bar during batch email prediction -->

<!-- Build step 48: Add export option for predictions to CSV file -->

<!-- Build step 49: Enhance logger formatting with timestamps and levels -->

<!-- Build step 50: Add file path validator in configuration script -->

<!-- Build step 51: Improve error messaging for invalid text input -->

<!-- Build step 52: Refactor TF-IDF vectorization parameter defaults -->

<!-- Build step 53: Update model hyperparameter tuning grid search -->

<!-- Build step 54: Add cross-validation fold configuration options -->

<!-- Build step 55: Create pipeline output directory structure automated creator -->

<!-- Build step 56: Optimize MBOX temporary file cleanup logic -->

<!-- Build step 57: Add sample dataset CSV file to repository -->

<!-- Build step 58: Include pre-trained vectorizer artifact in data models -->

<!-- Build step 59: Add trained model pickle artifact for version 1 -->

<!-- Build step 60: Create log storage directory configuration structure -->

<!-- Build step 61: Update requirements file with pinned dependency versions -->

<!-- Build step 62: Enhance README with usage instructions for batch processing -->

<!-- Build step 63: Add section on model re-training to README -->

<!-- Build step 64: Update pyproject metadata with author and version -->

<!-- Build step 65: Refactor logger import statement in app module -->

<!-- Build step 66: Add input length validation in Streamlit text input area -->

<!-- Build step 67: Improve spam vs ham label formatting in UI output -->

<!-- Build step 68: Add metric visualization charts in training pipeline -->

<!-- Build step 69: Implement model artifact serialization checks -->

<!-- Build step 70: Support custom model artifact loading path setting -->

<!-- Build step 71: Fix potential memory leak during MBOX batch iteration -->

<!-- Build step 72: Clean up unused helper functions in utils script -->

<!-- Build step 73: Add inline documentation docstrings across components -->

<!-- Build step 74: Standardize exception logging format across pipelines -->

<!-- Build step 75: Implement environment variable override for paths -->

<!-- Build step 76: Refactor dataset loading exception handling -->

<!-- Build step 77: Optimize regex patterns for email link extraction -->

<!-- Build step 78: Add unit test suite setup for preprocessor module -->

<!-- Build step 79: Implement feature extraction validation checks -->

<!-- Build step 80: Enhance Streamlit theme styling and icon display -->

<!-- Build step 81: Add dataset balance statistics printer in ingestion -->

<!-- Build step 82: Support configurable max features in vectorizer -->

<!-- Build step 83: Implement graceful fallback when model files missing -->

<!-- Build step 84: Optimize memory footprint during large text vectorization -->

<!-- Build step 85: Refactor prediction pipeline output dictionary schema -->

<!-- Build step 86: Add retry logic for file file-system operations -->

<!-- Build step 87: Update gitignore to exclude local log outputs -->

<!-- Build step 88: Enhance log level configuration support via config -->

<!-- Build step 89: Add parameter checks for model evaluation function -->

<!-- Build step 90: Standardize return types in email processing helpers -->

<!-- Build step 91: Optimize batch prediction speed using vectorized operations -->

<!-- Build step 92: Clean up temporary directory references post batch run -->

<!-- Build step 93: Add comprehensive comment header to entrypoint app -->

<!-- Build step 94: Refactor model loading caching decorator mechanism -->
