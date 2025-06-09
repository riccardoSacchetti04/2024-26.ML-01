import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import sklearn
from sklearn.metrics import (
    mean_absolute_error, 
    median_absolute_error, 
    mean_absolute_percentage_error,
    make_scorer,
    accuracy_score,
    recall_score, 
    f1_score,
    precision_score

)
import optuna 
from scipy.stats import randint
from xgboost import XGBClassifier
from category_encoders import LeaveOneOutEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders import BinaryEncoder
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

df = pd.read_csv('ml_datasets/heart_2022_no_nans.csv')


smoke_order = ['Never smoked', 'Former smoker', 'Current smoker - now smokes some days', 'Current smoker - now smokes every day']
df['SmokerStatus'] = df['SmokerStatus'].map({v: i for i, v in enumerate(smoke_order)}).astype('str')

health_order=['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
df['GeneralHealth'] = df['GeneralHealth'].map({v: i for i, v in enumerate(health_order)}).astype('str')

checkup_order=['Within past year (anytime less than 12 months ago)','Within past 2 years (1 year but less than 2 years ago)', 'Within past 5 years (2 years but less than 5 years ago)', '5 or more years ago']
df['LastCheckupTime'] = df['LastCheckupTime'].map({v: i for i, v in enumerate(checkup_order)}).astype('str')

teeth_order=['None of them', '1 to 5', '6 or more, but not all', 'All']
df['RemovedTeeth'] = df['RemovedTeeth'].map({v: i for i, v in enumerate(teeth_order)}).astype('str')

cigar_order= ['Never used e-cigarettes in my entire life', 'Not at all (right now)', 'Use them some days', 'Use them every day']
df['ECigaretteUsage'] = df['ECigaretteUsage'].map({v: i for i, v in enumerate(cigar_order)}).astype('str')

age_order=[
    'Age 18 to 24',
    'Age 25 to 29',
    'Age 30 to 34',
    'Age 35 to 39',
    'Age 40 to 44',
    'Age 45 to 49',
    'Age 50 to 54',
    'Age 55 to 59',
    'Age 60 to 64',
    'Age 65 to 69',
    'Age 70 to 74',
    'Age 75 to 79',
    'Age 80 or older'
]
df['AgeCategory'] = df['AgeCategory'].map({v: i for i, v in enumerate(age_order)}).astype('str')

groupBool=['Sex', 'PhysicalActivities', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
        'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'DeafOrHardOfHearing',
        'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking',
       'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers',
       'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'HighRiskLastYear', 'CovidPos']

groupOrder=['GeneralHealth', 'LastCheckupTime', 'RemovedTeeth', 'SmokerStatus', 'ECigaretteUsage', 'AgeCategory']

groupOther=['State', 'RaceEthnicityCategory', 'TetanusLast10Tdap', 'HadDiabetes', 'CovidPos']

df[groupBool] = df[groupBool].replace({'Yes': 1, 'No': 0})
df['Sex']=df['Sex'].replace({'Male': 1, 'Female': 0})
df['HadHeartAttack']=df['HadHeartAttack'].replace({'Yes': 1, 'No': 0})
df[['HadDiabetes', 'CovidPos']]=df[['HadDiabetes', 'CovidPos']].astype(str)

sklearn.set_config(transform_output='pandas')
x = df.drop(columns=['HadHeartAttack'])
y = df['HadHeartAttack']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

best_paramsRandom={'enc_bool': 'binary',
 'enc_ord': 'target',
 'ord__target_smoothing': 2.044188250637812,
 'enc_oth': 'onehot',
 'other__onehot_min_freq': 1,
 'n_estimators': 185,
 'max_depth': 5,
 'criterion': 'gini'}

# enc_bool = TargetEncoder(
#     smoothing=best_params['bool__target_smoothing']
# )
enc_bool = BinaryEncoder(

)
enc_ord = TargetEncoder(
    #handle_unknown='infrequent_if_exist'
    smoothing=best_paramsRandom['ord__target_smoothing']
)
enc_oth = OneHotEncoder (
    handle_unknown='infrequent_if_exist',
    sparse_output=False,
    min_frequency=best_paramsRandom['other__onehot_min_freq']
)


preprocessor = ColumnTransformer([
    ('bool', enc_bool, groupBool),
    ('ord', enc_ord, groupOrder),
    ('other', enc_oth, groupOther)

])
best_model = RandomForestClassifier(
    n_estimators=best_paramsRandom['n_estimators'],
    max_depth=best_paramsRandom['max_depth'],
    criterion=best_paramsRandom['criterion'],
    #class_weight=best_params.get('class_weight', None),
    random_state=42,
    class_weight='balanced'
)
pipe = Pipeline([
    ("preprocessor", preprocessor),  
    ("standardization", StandardScaler()),
    ("classifier", best_model)
])

pipe.fit(x_train, y_train)

joblib.dump(pipe, 'bestHeartPred.joblib')