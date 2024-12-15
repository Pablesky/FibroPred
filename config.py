import torch

SPLIT_CHAR = ';'

SKIP_COLUMNS = [
    'FVC (L) at diagnosis',
    'FVC (%) at diagnosis',
    'DLCO (%) at diagnosis',
    'FVC (L) 1 year after diagnosis',
    'FVC (%) 1 year after diagnosis',
    'DLCO (%) 1 year after diagnosis'
]

DATE_TARGET_COLUMN = 'Transplantation Status'
DEATH_TARGET_COLUMN = 'Death'

PRECISSION = torch.long

TARGET_COLUMNS = [
    'Progressive disease', 
    'Death', 
    'Necessity of transplantation'
]

DELETE_COLUMNS = ['COD NUMBER', 'Transplantation Status']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_DATA = 'content/FibroPredCODIFICADA_Updated_after_diagnosis.csv'
OUTPUT_DATA = 'content/data_prueba.csv'

AGE_COLUMN = 'Age_Group'

IMAGES_NAMES = [
    'Progressive disease', 
    'Death', 
    'Necessity of transplantation'
]

METRIC_COLUMN = 'Metric'