# Iris_flower_prediction
A Streamlit web app that predicts Iris flower species in real time from user-input measurements, powered by a KNN classifier trained and evaluated with cross-validation and hyperparameter tuning.
## Demo
See it in action 
[iris_flower.webm](https://github.com/user-attachments/assets/e2e5bd84-8d66-4582-a081-1b570afced5f)

## Dataset
load_iris(built in sklearn)
- 150 rows with 4 features and 3 classes, no missing values and duplicate/redundant columns
Consist of 150 samples

## Process
1. **Evaluation** - accuracy, confusion Matrix, classification report,overfitting checks(train vs test),     5-fold cross-validation
2. **Tuning** - GridSearchCV for model's key hyperparameter

### Model Training
I used KNN as the classification algorithm, chosen for its simplicity and strong performance on well-separated, low-dimensional dataset like Iris.
Before training, features were split into training and testing sets using a stratified split which preserves class proportions.
Since KNN relies on distance calculations, features were also scaled using StandardScaler which shows little difference for this particular dataset.
The model was first trained with n_neighbors=5, which later proved to 
be the optimal choice after hyperparameter tuning.
### Evaluation
- Model accuracy is 93.33%. The confusion matrix shows classes 0 and 1 were classified perfectly, while 2 samples from class 2 were misclassified as class 1. 
- Training accuracy (97.5%) and test accuracy is (93.3%) are close, showing no overfitting - the model generalizes well rather than memorizing training data.
- A single train/test split can be misleading on a small dataset, so I also ran 5-fold cross-validation for a more reliable estimate:
   - Scaled KNN CV average: 96.0% (scores: 0.97, 0.97, 0.93, 0.93, 1.00)
   - Unscaled KNN CV average: 97.3%
Both are very close, confirmig that feature scaling has little effect on this dataset.
### Hyperparameter Tuning
Used GridSearchCV to test n_neighbors values from 3 to 13. The best result was n_neighbors=5 with a cross validated accuracy of 96%, confirming the default choice used during initial training was already optimal.

## Error Handling & Input Validation
1- try/except catching missing model files with a clean error message
2- input validation rejecting negative values and out-of-realistic-range measurements, showing all errors at once.
<img width="1801" height="1022" alt="errors" src="https://github.com/user-attachments/assets/6e2b4f6d-990c-4a78-bd23-ee1803be3fcf" />


## Results
| Metric | Score |
|---|---|
| Test Accuracy (single split) | 93.3% |
| Training Accuracy | 97.5% |
| Cross-Validation Average (scaled) | 96.0% |
| Cross-Validation Average (unscaled) | 97.3% |
| Best Hyperparameter (GridSearchCV) | n_neighbors=5 |
| Best CV Score (GridSearchCV) | 96.0% |

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Jupyter Notebook
- Streamlit
- CSS

## Project Structure
```
├── app.py
├── iris_model.pkl
├── iris_scaler.pkl
├── iris_flower_prediction.ipynb  (use your real filename)
```
## How to Run
1. Clone the repo
2. Install dependencies: `pip install streamlit scikit-learn joblib numpy`
3. Run the app: 'streamlit run app.py'

## Key Learnings
- A single train/test split can be misleading, especially on small 
  datasets — one lucky or unlucky split can make a model look better 
  or worse than it really is. Cross-validation gives a much more 
  honest, stable estimate of true performance.
- Real applications need more than just a working model — building 
  proper input validation and error handling (missing files, unrealistic 
  input values) made the difference between a notebook exercise and 
  something that behaves like a real, usable tool.
- Converting a trained model into an actual interactive app (via 
  joblib + Streamlit) is a distinct skill from training the model 
  itself — required understanding how to save/load models, and keeping 
  the scaler consistent between training and live predictions.
