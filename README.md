# MLZoomcamp 2022 Midterm project

# Heart Failure Prediction

## Description:

Cardiovascular problems are the number 1 cause of death globally, the majority of fatal cardiovascular diseases are heart attacks and strokes and some of this health problems can occurs in individuals under 70 years old. Heart failure is a common event caused by cardiovascular diseases and can be predicted using some patient information. Then, classification machine learning models could be used to help doctors to detect this kind of event in patients with cardiovascular diseases or with high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease).

In order to do this task, this project offers a classification model trained using the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction), where it has 11 columns with medical and personal information that can be used to predict the probability of a heart failure in a given patient.


## How to run the application:

### 1. Docker

Step 1: Clone the GitHub repository with the project.
```bash
git clone https://github.com/jvscursulim/mlzoom2022_midterm_project
```

Step 2: Access the GitHub repository folder.
```bash
cd mlzoom2022_midterm_project
```

Step 3: Create the docker image.
```bash
docker build -t heart_failure_prediction .
```

Step 4: Run the application with docker.
```bash
docker run -p 4242:4242 heart_failure_prediction
```

### 2. Without Docker

Step 1: Clone the GitHub repository with the project.
```bash
git clone https://github.com/jvscursulim/mlzoom2022_midterm_project
```

Step 2: Access the GitHub repository folder.
```bash
cd mlzoom2022_midterm_project
```

Step 3: Create a virtual environment.
```bash
python -m venv env
```

Step 4: Activate your virtual environment.
* Linux: Activation of the virtual environment.
```bash
source env/bin/activate
```

* Windows: Activation of the virtual environment.
```bash
env/Scripts/Activate.ps1
```

Step 5: Install pipenv.
```bash
pip install pipenv
```

Step 6: Install the packages required for this application using the command below.
```bash
pipenv install
```

Step 7: Run the application with gunicorn.
```bash
gunicorn --bind=0.0.0.0:4242 predict:app
```

#### Observation: If you want to train a model

Access script folder.
```bash
cd script
```
Make your changes in `train.py` and run the file using the command below.
```bash
python train.py
```
After training process, you can build the new application following the instructions in the sections 1. Docker and 2. Without Docker.

## How to send data for the application:

### Code snippet

```python
import requests

patient = {'age': 47.0,
           'sex': 0.0,
           'chest_pain_type': 1.0,
           'resting_bp': 130.0,
           'cholesterol': 235.0,
           'fasting_bs': 0.0,
           'resting_ecg': 0.0,
           'max_hr': 145.0,
           'exercise_angina': 0.0,
           'oldpeak': 2.0,
           'st_slope': 0.0}

url = "http://localhost:4242/predict"
print(requests.post(url, json=patient).json())
```

## References:

[1] Alexey Grigorev, "Machine Learning Zoomcamp". https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp

[2] fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [Date Retrieved] from https://www.kaggle.com/fedesoriano/heart-failure-prediction.

