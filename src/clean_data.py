import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Drop non-predictive columns
    df.drop(columns=[
        'id','country','diagnosis_date',
        'end_treatment_date','treatment_type'
    ], inplace=True)

    # Handle missing values
    df['cancer_stage'] = df['cancer_stage'].fillna(df['cancer_stage'].mode()[0])

    # Encode categorical values
    df['family_history'] = df['family_history'].map({'Yes':1,'No':0})
    df['gender'] = df['gender'].map({'Male':1,'Female':0})

    df['smoking_status'] = df['smoking_status'].map({
        'Never Smoked':0,
        'Passive Smoker':1,
        'Former Smoker':2,
        'Current Smoker':3
    })

    df['cancer_stage'] = df['cancer_stage'].map({
        'Stage I':1,
        'Stage II':2,
        'Stage III':3,
        'Stage IV':4
    })

    # Scale numerical features
    scaler = StandardScaler()
    df[['age','bmi','cholesterol_level']] = scaler.fit_transform(
        df[['age','bmi','cholesterol_level']]
    )

    # Save cleaned data
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_data(
        "../data/raw/lung_cancer_raw.csv",
        "../data/processed/lung_cancer_cleaned.csv"
    )
