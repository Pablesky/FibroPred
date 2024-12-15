import pandas as pd
from datetime import datetime

# READ DATA
excel_data = pd.ExcelFile("FibroPredCODIFICADA.xlsx")
data = excel_data.parse('Hoja1')
data.columns = data.iloc[0]
data = data.drop(0).reset_index(drop=True)

# DERIVE COLUMNS
# 1. Create age group based on the age at diagnosis
def categorize_age(age):
    if age < 36:
        return '< 36'
    elif 36 <= age <= 55:
        return '36-55'
    elif 56 <= age <= 65:
        return '56-65'
    elif 66 <= age <= 75:
        return '66-75'
    else:
        return '76+'

data['Age_Group'] = data['Age at diagnosis'].apply(categorize_age)

# 2. Group "UIP" and "Probable UIP" in Radiological Pattern
data['Radiological Pattern'] = data['Radiological Pattern'].replace({'Probable UIP': 'UIP'})

# 3. Create a new column "Antiinflmatory Drug"
data['Antiinflmatory Drug'] = ((data['Prednisone'] == 1) | (data['Mycophenolate'] == 1)).astype(int)

# 4. Create a new column with the subtraction of DLCO (%) 1 year after diagnosis and DLCO (%) at diagnosis
data['DLCO (%) 1 year after diagnosis'] = pd.to_numeric(data['DLCO (%) 1 year after diagnosis'], errors='coerce')
data['DLCO (%) at diagnosis'] = pd.to_numeric(data['DLCO (%) at diagnosis'], errors='coerce')
data['DLCO (%) Change'] = data['DLCO (%) 1 year after diagnosis'] - data['DLCO (%) at diagnosis']
data['Significant DLCO (%) Change'] = (data['DLCO (%) Change'] > 10).astype(int)

# 6. Create a new column with the subtraction of FVC (%) 1 year after diagnosis and FVC (%) at diagnosis
data['FVC (%) 1 year after diagnosis'] = pd.to_numeric(data['FVC (%) 1 year after diagnosis'], errors='coerce')
data['FVC (%) at diagnosis'] = pd.to_numeric(data['FVC (%) at diagnosis'], errors='coerce')
data['FVC (%) Change'] = data['FVC (%) 1 year after diagnosis'] - data['FVC (%) at diagnosis']
data['Significant FVC (%) Change'] = (data['FVC (%) Change'] > 5).astype(int)

# 7. Create a new column with the subtraction of FVC (L) 1 year after diagnosis and FVC (L) at diagnosis
data['FVC (L) 1 year after diagnosis'] = pd.to_numeric(data['FVC (L) 1 year after diagnosis'], errors='coerce')
data['FVC (L) at diagnosis'] = pd.to_numeric(data['FVC (L) at diagnosis'], errors='coerce')
data['FVC (L) Change'] = data['FVC (L) 1 year after diagnosis'] - data['FVC (L) at diagnosis']


# CHECK COLUMNS
# 1. Update Final Diagnosis based on Binary Diagnosis
data.loc[data['Binary diagnosis'] == 'IPF', 'Final diagnosis'] = 1

# TODO: bipsy, pathology pattern binary and diagnosis after biopsy

# 3. Update Extrapulmonary affectation based on Type of telomeric extrapulmonary affectation
data.loc[(data['Extrapulmonary affectation'] == 0) & (data['Type of telomeric extrapulmonary affectation'].notnull()), 'Extrapulmonary affectation'] = 1

# 4. Update Hematologic Disease based on its value
data.loc[(data['Hematologic Disease'].notnull()) & (data['Hematologic Disease'].str.strip().str.lower() != 'no'), 'Hematologic Disease'] = 'Yes'

# 5. Update Liver disease based on its value
data.loc[(data['Liver disease'].notnull()) & (data['Liver disease'].str.strip().str.lower() != 'no'), 'Liver disease'] = 'Yes'

# 6. Create a new column based on Transplantation date
data['Transplantation Status'] = data['Transplantation date'].apply(lambda x: 'Yes' if isinstance(x, datetime) else (None if pd.isnull(x) else 'No'))

# 7. Update Death status based on the date of death
data.loc[(data['Death'].str.strip().str.lower() == 'no') & (data['Date of death'].notnull()) & (data['Date of death'].str.strip().str.lower() != 'not dead'), 'Death'] = 'Yes'

# 8. Update Mutation Type based on presence
data['Mutation Type'] = data['Mutation Type'].apply(lambda x: 'Yes' if pd.notnull(x) and isinstance(x, str) and x.strip() != '' else 'No')

# 9. Convert Yes/No columns to binary values (1/0)
yes_no_columns = data.select_dtypes(include=['object']).columns
for col in yes_no_columns:
    if set(data[col].dropna().astype(str).str.lower().unique()) <= {'yes', 'no'}:
        data[col] = data[col].astype(str).str.lower().map({'yes': 1, 'no': 0})

# REMOVE COLUMNS
data.drop(columns=['Pedigree', 'Age at diagnosis', 'Detail', 'Detail on NON UIP', 'Pathology pattern UIP, probable or CHP',
                   'Pathology pattern', 'Extras AP', 'Multidsciplinary committee', 'Pirfenidone', 'Nintedanib',
                   'Prednisone', 'Mycophenolate', 'Treatment', 'Type of telomeric extrapulmonary affectation', 'Extra',
                   'Type of neoplasia', 'Type of liver abnormality', 'Transplantation date', 'Cause of death',
                   'Identified Infection', 'Date of death', 'Type of family history', 'Severity of telomere shortening - Transform 4',
                   'ProgressiveDisease'], inplace=True)

# Save the updated file after 2 years
data.to_csv('FibroPredCODIFICADA_Updated_after_2years.csv', index=False, sep=';')

# Remove specific columns and save after 1 year
data.drop(columns=['RadioWorsening2y'], inplace=True)
data.to_csv('FibroPredCODIFICADA_Updated_after_1years.csv', index=False, sep=';')

# Remove additional calculated columns and save after diagnosis
data.drop(columns=['DLCO (%) Change','Significant DLCO (%) Change','FVC (%) Change','Significant FVC (%) Change',
                   'FVC (L) Change', 'DLCO (%) 1 year after diagnosis', 'FVC (%) 1 year after diagnosis',
                   'FVC (L) 1 year after diagnosis',], inplace=True)
data.to_csv('FibroPredCODIFICADA_Updated_after_diagnosis.csv', index=False, sep=';')