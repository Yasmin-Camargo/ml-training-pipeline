from config import TARGET_COLUMN
from logger import log_message

def _process_model_A(df):
    """
    Model A:
    - Não filtra linhas.
    - Labels: (0 ou 1) viram 10, o resto vira 20.
    """
    df_out = df.copy()
    
    # Lógica de transformação de label (inline)
    df_out[TARGET_COLUMN] = df_out[TARGET_COLUMN].apply(lambda x: 10 if x in [0, 1] else 20)
    
    return df_out

def _process_model_B(df):
    """
    Model B:
    - Remove linhas onde Target é 0 ou 1.
    - Labels: (2) vira 10, o resto vira 20.
    """
    # Lógica de filtro (inline)
    df_out = df[~df[TARGET_COLUMN].isin([0, 1])].copy()
    
    # Lógica de transformação de label (inline)
    df_out[TARGET_COLUMN] = df_out[TARGET_COLUMN].apply(lambda x: 10 if x == 2 else 20)
    
    return df_out

def _process_logistic_raw(df):
    """
    Model C:
    - Dados brutos (sem filtro, sem renomear labels).
    """
    return df.copy()



"""
MODEL_DEFINITIONS = {
    "logistic_regression": {
        "description": "Model C (Logistic Regression - Raw Data)",
        "process_function": _process_logistic_raw,
        "classifier_name": "logistic_regression"
    }
}
"""
MODEL_DEFINITIONS = {
    "A": {
        "description": "Model A (Filter after DCT2)",
        "process_function": _process_model_A,
        "classifier_name": "decision_tree"
    },
    "B": {
        "description": "Model B (DST7 vs Others)",
        "process_function": _process_model_B,
        "classifier_name": "decision_tree"
    }
}


# --- Estratégias de Agrupamento de Blocos (Mantido igual) ---
def determine_size_group(row):
    w = row["Width"]
    h = row["Height"]

    max_dim = max(w, h)
    if max_dim == 128:
        return "128×128"
    elif max_dim == 64:
        return "64×64"
    elif max_dim == 32:
        return "32×32"
    elif max_dim == 16:
        return "16×16"
    elif max_dim == 8:
        return "8×8"
    else:
        return "4×4"


def determine_area_group(row):
    w = row["Width"]
    h = row["Height"]

    area_to_group = {
        16: "G0", 32: "G1", 64: "G2", 128: "G3",
        256: "G4", 512: "G5", 1024: "G6", 2048: "G7",
        4096: "G8", 8192: "G9", 16384: "G10"
    }
    area = min(w, h) * max(w, h)
    return area_to_group.get(area, "other")


def determine_all_group(row):
    w = row["Width"]
    h = row["Height"]

    return f"{w}×{h}"


def determine_orientation_group(row):
    w = row["Width"]
    h = row["Height"]

    if w == h:
        return "Square"
    elif w > h:
        return "Horizontal"
    else:
        return "Vertical"


def determine_aspect_ratio_group(row):
    w = row["Width"]
    h = row["Height"]

    ratio = max(w, h) / min(w, h)

    if abs(ratio - 1) < 0.01:
        return "1:1"
    elif abs(ratio - 2) < 0.01:
        return "2:1"
    elif abs(ratio - 4) < 0.01:
        return "4:1"
    elif abs(ratio - 8) < 0.01:
        return "8:1"
    elif abs(ratio - 16) < 0.01:
        return "16:1"
    elif abs(ratio - 32) < 0.01:
        return "32:1"
    else:
        return "other"


def determine_single_group(row):
    return "All_Blocks"

def apply_grouping(df, grouping_name, grouping_strategies):
    log_message(f"Applying grouping strategy: {grouping_name}")
    
    if grouping_name not in grouping_strategies:
        raise ValueError(f"Grouping strategy '{grouping_name}' not found.")
    
    grouping_func = grouping_strategies[grouping_name]
    
    df = df.copy()
    df["BlockGroup"] = df.apply(grouping_func, axis=1)
    
    return df, sorted(df["BlockGroup"].unique())



# MAPA DE ESTRATÉGIAS DISPONÍVEIS
# Mapeia uma string (nome) para a função correspondente
GROUPING_STRATEGIES = {
    'area': determine_area_group,
    'max': determine_size_group,
    'orientation': determine_orientation_group,
    'aspect_ratio': determine_aspect_ratio_group,
    'all': determine_all_group,
    'single': determine_single_group
}