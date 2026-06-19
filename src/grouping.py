import pandas as pd
from src.utils import log_message

def _group_by_frame_level(df):
    """
    Estratégia: Hierarquia Exaustiva
    Agrupa separando cada FrameLevel individualmente (1, 2, 3, 4, 5).
    """
    groups = {}
    for level in df['FrameLevel'].unique():
        groups[str(int(level))] = df[df['FrameLevel'] == level].copy()
    return groups

def _group_by_frame_tier(df):
    """
    Estratégia: Hierarquia Otimizada (Base vs Descartáveis)
    Base_Tier: Níveis 1 e 2 (Imagens de Referência / Alta qualidade)
    Leaf_Tier: Níveis 3, 4, 5 (Imagens não-referência / Fortemente comprimidas)
    """
    groups = {}
    groups['Base_Tier'] = df[df['FrameLevel'] <= 2].copy()
    groups['Leaf_Tier'] = df[df['FrameLevel'] > 2].copy()
    return groups

def _group_by_neighborhood(df):
    """
    Estratégia: Contexto Espacial (Baseado na sua melhor feature)
    Pure_Inter_Context: Nenhum vizinho fez Intra (num_intra_ciip_neighbors == 0)
    Mixed_Intra_Context: Pelo menos 1 vizinho fez Intra (num_intra_ciip_neighbors > 0)
    """
    groups = {}
    groups['Pure_Inter_Context'] = df[df['num_intra_ciip_neighbors'] == 0].copy()
    groups['Mixed_Intra_Context'] = df[df['num_intra_ciip_neighbors'] > 0].copy()
    return groups

def _group_by_block_size(df):
    """
    Estratégia: Normalização de Custo Inter (Isola a matemática do inter_cost)
    Large_Blocks: Área >= 64x64 (BlockAreaGroup >= 6)
    Medium_Blocks: Área 16x16 a 32x32 (BlockAreaGroup 4 e 5)
    Small_Blocks: Área <= 8x8 (BlockAreaGroup <= 3)
    """
    groups = {}
    groups['Large_Blocks'] = df[df['BlockAreaGroup'] >= 6].copy()
    groups['Medium_Blocks'] = df[(df['BlockAreaGroup'] == 4) | (df['BlockAreaGroup'] == 5)].copy()
    groups['Small_Blocks'] = df[df['BlockAreaGroup'] <= 3].copy()
    return groups

def _group_by_qp(df):
    """
    Estratégia: Agressividade da Compressão
    High_Quality_QP: TargetQP < 30 (Muitos detalhes mantidos)
    Low_Quality_QP: TargetQP >= 30 (Forte compressão/borrão)
    """
    groups = {}
    groups['High_Quality_QP'] = df[df['TargetQP'] < 30].copy()
    groups['Low_Quality_QP'] = df[df['TargetQP'] >= 30].copy()
    return groups

def _group_by_texture(df):
    """
    Estratégia: Complexidade Espacial (Fáceis vs Difíceis)
    Flat_Texture: Variância abaixo da mediana (blocos fáceis/lisos, ex: céu)
    Complex_Texture: Variância acima da mediana (blocos caóticos, ex: folhas)
    """
    groups = {}
    median_var = df['blk_pixel_variance'].median()
    groups['Flat_Texture'] = df[df['blk_pixel_variance'] <= median_var].copy()
    groups['Complex_Texture'] = df[df['blk_pixel_variance'] > median_var].copy()
    return groups

# --- Grouping Logic Functions ---

def determine_size_group(row):
    w = row["BlockWidth"]
    h = row["BlockHeight"]
    max_dim = max(w, h)
    
    if max_dim == 128: return "128x128"
    elif max_dim == 64: return "64x64"
    elif max_dim == 32: return "32x32"
    elif max_dim == 16: return "16x16"
    elif max_dim == 8: return "8x8"
    else: return "4x4"

def determine_area_group(row):
    w = row["BlockWidth"]
    h = row["BlockHeight"]
    area = min(w, h) * max(w, h)
    
    area_to_group = {
        16: "G0", 32: "G1", 64: "G2", 128: "G3",
        256: "G4", 512: "G5", 1024: "G6", 2048: "G7",
        4096: "G8", 8192: "G9", 16384: "G10"
    }
    return area_to_group.get(area, "other")

def determine_all_group(row):
    w = row["BlockWidth"]
    h = row["BlockHeight"]
    return f"{w}x{h}"

def determine_orientation_group(row):
    w = row["BlockWidth"]
    h = row["BlockHeight"]
    if w == h: return "Square"
    elif w > h: return "Horizontal"
    else: return "Vertical"

def determine_aspect_ratio_group(row):
    w = row["BlockWidth"]
    h = row["BlockHeight"]
    ratio = max(w, h) / min(w, h)

    if abs(ratio - 1) < 0.01: return "1:1"
    elif abs(ratio - 2) < 0.01: return "2:1"
    elif abs(ratio - 4) < 0.01: return "4:1"
    elif abs(ratio - 8) < 0.01: return "8:1"
    elif abs(ratio - 16) < 0.01: return "16:1"
    elif abs(ratio - 32) < 0.01: return "32:1"
    else: return "other"

def determine_single_group(row):
    return "All_Blocks"

def determine_frame_level_group(row):
    level = row["FrameLevel"]
    
    if level == 0:
        return None
        
    return int(level)

# --- Strategy Map ---
GROUPING_STRATEGIES = {
    'area': determine_area_group,
    'max': determine_size_group,
    'orientation': determine_orientation_group,
    'aspect_ratio': determine_aspect_ratio_group,
    'all': determine_all_group,
    'single': determine_single_group,
    'frame_level': _group_by_frame_level,
    'frame_tier': _group_by_frame_tier,
    'neighborhood': _group_by_neighborhood,
    'block_size': _group_by_block_size,
    'qp': _group_by_qp,
    'texture': _group_by_texture
}

def apply_grouping_strategy(df, strategy_name):
    if strategy_name not in GROUPING_STRATEGIES:
        raise ValueError(f"Grouping strategy '{strategy_name}' not found.")
    
    log_message(f"Applying grouping strategy: {strategy_name}", level="INFO")
    
    df_out = df.copy()
    
    df_out["BlockGroup"] = df_out.apply(GROUPING_STRATEGIES[strategy_name], axis=1)
    
    before_drop = len(df_out)
    df_out = df_out.dropna(subset=["BlockGroup"])
    after_drop = len(df_out)
    
    if before_drop > after_drop:
        log_message(f"Strategy '{strategy_name}' filtered out {before_drop - after_drop} rows (e.g. FrameLevel 0).", level="WARNING")
    
    return df_out, sorted(df_out["BlockGroup"].unique())
