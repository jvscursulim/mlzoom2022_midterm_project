# imports
import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# loading data
df = pd.read_csv("../data/heart.csv")

# a list with new columns names
new_columns_names = ["age",
                     "sex",
                     "chest_pain_type",
                     "resting_bp",
                     "cholesterol",
                     "fasting_bs",
                     "resting_ecg",
                     "max_hr",
                     "exercise_angina",
                     "oldpeak",
                     "st_slope",
                     "heart_disease"]

# changing columns names
df.columns = new_columns_names

# map for sex column
sex_map = {"F": 0,
           "M": 1}

# applying the map for sex column and changing its dtype
df.sex = df.sex.map(sex_map)
df.sex = df.sex.astype(int)

# map for chest_pain_type
chest_pain_type_map = {"ASY": 0,
                       "NAP": 1,
                       "ATA": 2,
                       "TA": 3}

# applying the map for chest_pain_type column and changing its dtype
df.chest_pain_type = df.chest_pain_type.map(chest_pain_type_map)
df.chest_pain_type = df.chest_pain_type.astype(int)

# map for resting_ecg column
resting_ecg_map = {"Normal": 0,
                   "LVH": 1,
                   "ST": 2}

# applying the map for resting_ecg column and changing its dtype
df.resting_ecg = df.resting_ecg.map(resting_ecg_map)
df.resting_ecg = df.resting_ecg.astype(int)

# map for exercise_angina column
exercise_angina_map = {"N": 0,
                       "Y": 1}

# applying the map for exercise_angina column and changing its dtype
df.exercise_angina = df.exercise_angina.map(exercise_angina_map)
df.exercise_angina = df.exercise_angina.astype(int)

# map for st_slop column
st_slop_map = {"Flat": 0,
               "Up": 1,
               "Down": 2}

# applying the map for st_slope column and changing its dtype
df.st_slope = df.st_slope.map(st_slop_map)
df.st_slope = df.st_slope.astype(int)

# removing some inconsistent data in cholesterol and resting_bp columns
df = df[~(df.cholesterol == 0)].copy()
df = df[~(df.resting_bp == 0)].copy()

# creating a new variable to store a copy of the DataFrame
data = df.copy()

# defining the target column and the features that will be used in the model
TARGET = "heart_disease"
FEATURES = ["exercise_angina", 
            "oldpeak", 
            "age", 
            "sex", 
            "chest_pain_type", 
            "st_slope", 
            "fasting_bs"]

# using train_test_split to prepare the data for training and test
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(data[FEATURES],
                                                    data[TARGET],
                                                    test_size=test_size,
                                                    random_state=42)

# initializing DictVectorizer
dv = DictVectorizer(sparse=False)

# converting X_train and X_test for dictionaries
train_dict = X_train.to_dict(orient="records")
test_dict = X_test.to_dict(orient="records")

# creating a new X_train and X_test from the dictionaries using DictVectorizer method called fit_transform.
X_train = dv.fit_transform(train_dict)
X_test = dv.fit_transform(test_dict)

# training the DecisionTreeClassfier
dt = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=1, random_state=42)
dt.fit(X=X_train, y=y_train)

# saving the model in a binary file using pickle
with open("../model/model.bin", "wb") as model_file:
    
    pickle.dump(dt, model_file)

# saving the DictVectorizer in a binary file using pickle    
with open("../model/dv.bin", "wb") as dv_file:
    
    pickle.dump(dv, dv_file)


