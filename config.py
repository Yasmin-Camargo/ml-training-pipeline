
# Configurações gerais
#file_path = './features_from_image.csv'
file_path = './features_10000_por_classe.csv'
is_validation_curve = False
export_decision_tree_cpp = False

# Separador do CSV
#CSV_SEPARATOR = ';'
CSV_SEPARATOR = ','

# Nome da coluna alvo (target)
#TARGET_COLUMN = 'IsIntra'
TARGET_COLUMN = 'MTSChosen'

# Colunas para balanceamento
#BALANCE_COLUMNS = ['IsIntra', 'TargetQP', 'FrameWidth', 'FrameHeight']
BALANCE_COLUMNS = ['MTSChosen', 'cuQP', 'FrameWidth', 'FrameHeight']

# Colunas a serem removidas no pré-processamento
#REMOVE_COLUMNS = ['VideoName', 'EncoderPreset', 'BlockAreaGroup']
REMOVE_COLUMNS = ['VideoName']


# CONFIGURAÇÃO DE EXECUÇÃO
# O usuário edita AQUI para escolher quais rodar.
# Se deixar a lista vazia [], nenhum loop de agrupamento roda (ou gera erro, dependendo da lógica).
ACTIVE_GROUPINGS = ['single']


TEST_SIZE = 0.25    
RANDOM_STATE = 42
NUMBER_OF_JOBS = -1
MAX_SAMPLES_PER_CLASS = 100000
CROSS_VALIDATION = 5
SCORING = 'accuracy'

# Configurações do DecisionTreeClassifier
DTREE_CRITERION = ['gini', 'entropy', 'log_loss']
DTREE_MIN_SAMPLES_SPLIT = list(range(2, 100, 10))
DTREE_MIN_SAMPLES_LEAF = list(range(2, 200, 10))
DTREE_MAX_LEAF_NODES = list(range(10, 500, 25))

# Configurações do RFECV
RFECV_CV = CROSS_VALIDATION
RFECV_SCORING = SCORING
RFECV_STEP = 1
RFECV_MIN_FEATURES_SELECT = 5

# Configurações do Random Search
RANDOM_SEARCH_ITERATIONS = 100

# Configurações de validação
VAL_CURVE_PARAMS = {
    "max_depth": list(range(2, 10)) + list(range(10, 101, 10)),
    "min_samples_split": list(range(2, 25)) + list(range(25, 1001, 25)),
    "min_samples_leaf": list(range(2, 25)) + list(range(25, 1001, 25)),
    "max_leaf_nodes": list(range(2, 25)) + list(range(25, 1000, 25)),
}
VAL_SCORING = SCORING
VAL_CURVE_CV = CROSS_VALIDATION
VAL_CURVE_SCORING = SCORING
