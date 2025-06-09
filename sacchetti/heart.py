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

def getEncoder(trial, group):
    if group == 'bool':
        encoder = trial.suggest_categorical('enc_bool', ['onehot', 'binary', 'target'])
    
    elif group == 'ord':
        encoder = trial.suggest_categorical('enc_ord', ['ordinal', 'onehot', 'target'])
    
    elif group == 'other':
        encoder = trial.suggest_categorical('enc_oth', ['target', 'onehot'])

    if encoder == "onehot":
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
            min_frequency=trial.suggest_int(f"{group}__onehot_min_freq", 1, 20)
        )
    
    elif encoder == "ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            encoded_missing_value=-1,
            unknown_value=-1 
        )
        
    
    elif encoder == "binary":
        return BinaryEncoder()
    
    elif encoder == "target":
        return TargetEncoder(
            smoothing=trial.suggest_float(f"{group}__target_smoothing", 0.5, 12.0)
        )
    else:
        raise ValueError("Unsupported encoder type")


from sklearn.utils import resample
x_small, y_small = resample(x_train, y_train, n_samples=30000, random_state=42)


def objective(trial):
    
    preprocessor = ColumnTransformer([
        ("bool", getEncoder(trial, "bool"), groupBool),
        ("ord", getEncoder(trial, "ord"), groupOrder),
        ("other", getEncoder(trial, "other"), groupOther),
    ], remainder="drop")

    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 5, 40),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        random_state=42,
        class_weight='balanced'
    )

    pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('standardization', StandardScaler()),
    ('regressor', model)]
)
    values = cross_validate(
        pipe, 
        x_small, y_small,
        scoring={
            'recall' : make_scorer(recall_score, pos_label=1),
            'precision' : make_scorer(precision_score, pos_label = 1)
        },
        cv = KFold(shuffle=True, random_state=42, n_splits=8,)
    )

    recall = values["test_recall"].mean()
    # precision = values["test_precision"].mean()
    print(f"Trial {trial.number} - Recall: {recall:.4f} - Params: {trial.params}")
    return recall

studyHeartRandom = optuna.create_study(storage="sqlite:///model_selection.db", study_name="studyHeartDefinitive", direction="maximize" , load_if_exists=True)
studyHeartRandom.optimize(objective, n_trials=15, show_progress_bar=True)
joblib.dump(studyHeartRandom, 'bestHeartPred.joblib')