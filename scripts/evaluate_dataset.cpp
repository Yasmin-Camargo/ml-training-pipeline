#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <iomanip>
#include <algorithm>
#include <clocale>

// Inclua os cabeçalhos das suas árvores geradas
#include "decision_tree_single_mdecision_tree_All_Blocks-intrakept.h"
#include "decision_tree_single_mdecision_tree_All_Blocks-issplit.h"

// Estrutura para armazenar as métricas
struct VideoStats {
    int TP_intra = 0, TN_intra = 0, FP_intra = 0, FN_intra = 0, total_intra = 0;
    int TP_split = 0, TN_split = 0, FP_split = 0, FN_split = 0, total_split = 0;
};

// Função de mapeamento dos vídeos para as Classes CTC do VVC
std::string getVideoClass(const std::string& video_name) {
    static std::map<std::string, std::string> class_map = {
        // --- 4K (CLASS A1) ---
        {"Campfire", "A1"}, {"FoodMarket4", "A1"}, {"Tango2", "A1"},
        // --- 4K (CLASS A2) ---
        {"CatRobot", "A2"}, {"DaylightRoad2", "A2"}, {"ParkRunning3", "A2"},
        // --- 1080p (CLASS B) ---
        {"BasketballDrive", "B"}, {"MarketPlace", "B"}, {"BQTerrace", "B"}, {"Cactus", "B"}, {"RitualDance", "B"},
        // --- 480p (CLASS C) ---
        {"BQMall", "C"}, {"BasketballDrill", "C"}, {"RaceHorsesC", "C"}, {"PartyScene", "C"},
        // --- 240p (CLASS D) ---
        {"BQSquare", "D"}, {"BlowingBubbles", "D"}, {"BasketballPass", "D"}, {"RaceHorses", "D"},
        // --- 720p (CLASS E) ---
        {"FourPeople", "E"}, {"Johnny", "E"}, {"KristenAndSara", "E"},
        // --- (CLASS F) ---
        {"BasketballDrillText", "F"}, {"ArenaOfValor", "F"}, {"SlideEditing", "F"}, {"SlideShow", "F"}
    };

    auto it = class_map.find(video_name);
    if (it != class_map.end()) {
        return it->second;
    }
    return "Outros"; // Caso algum vídeo do dataset não esteja mapeado acima
}

// Função auxiliar para imprimir os relatórios dinamicamente
void print_report(const std::string& model_name, const std::string& label_0, const std::string& label_1, int TP, int TN, int FP, int FN, int total) {
    double accuracy = (double)(TP + TN) / total * 100.0;
    
    double precision_0 = (TN + FN) > 0 ? (double)TN / (TN + FN) * 100.0 : 0;
    double recall_0    = (TN + FP) > 0 ? (double)TN / (TN + FP) * 100.0 : 0;
    double f1_0        = (precision_0 + recall_0) > 0 ? 2 * (precision_0 * recall_0) / (precision_0 + recall_0) : 0;

    double precision_1 = (TP + FP) > 0 ? (double)TP / (TP + FP) * 100.0 : 0;
    double recall_1    = (TP + FN) > 0 ? (double)TP / (TP + FN) * 100.0 : 0;
    double f1_1        = (precision_1 + recall_1) > 0 ? 2 * (precision_1 * recall_1) / (precision_1 + recall_1) : 0;

    std::cout << "\n=============================================" << std::endl;
    std::cout << "  RESULTADOS: " << model_name << std::endl;
    std::cout << "=============================================\n" << std::endl;
    
    std::cout << "--- Matriz de Confusao ---" << std::endl;
    std::cout << "                   | Predito: 0 | Predito: 1" << std::endl;
    std::cout << "-------------------|------------|------------" << std::endl;
    std::cout << std::left << std::setw(18) << ("Real " + label_0 + ":") << " | " << std::right << std::setw(10) << TN << " | " << std::setw(10) << FP << std::endl;
    std::cout << std::left << std::setw(18) << ("Real " + label_1 + ":") << " | " << std::right << std::setw(10) << FN << " | " << std::setw(10) << TP << std::endl;
    std::cout << "\n";

    std::cout << "--- Relatorio de Desempenho ---" << std::endl;
    std::cout << "Acuracia Global: " << std::fixed << std::setprecision(2) << accuracy << "%\n" << std::endl;
    std::cout << "[" << label_0 << " (0)] -> Precision: " << precision_0 << "% | Recall: " << recall_0 << "% | F1: " << f1_0 << "%" << std::endl;
    std::cout << "[" << label_1 << " (1)] -> Precision: " << precision_1 << "% | Recall: " << recall_1 << "% | F1: " << f1_1 << "%\n" << std::endl;
}

int main() {
    setlocale(LC_NUMERIC, "C");

    // FEATURES INTRAKEPT
    std::string target_col_intra = "IntraKept"; 
    std::vector<std::string> features_intra = {
        "FrameLevel", "SplitSeries", "inter_had_per_pixel", "ref_line_range", 
        "num_intra_ciip_neighbors", "left_depth", "relative_block_area", 
        "delta_qp", "contrast_ratio", "directional_dominance", 
        "var_mismatch", "blk_std_v", "blk_range"
    };

    // FEATURES ISSPLIT
    std::string target_col_split = "IsSplit"; 
    std::vector<std::string> features_split = {
        "BlockWidth", "inter_cost", "ref_col_variance", "above_depth", 
        "neighbor_mean_depth", "delta_qp", "variance_per_area", 
        "dist_center_y", "splitting_density", "center_focus_weight"
    };

    std::string video_col_name = "VideoName";

    // Abertura do Arquivo
    std::ifstream file("/home/yasminsc/ml-training-pipeline/data/dataset_ctc_vvc_intra.csv");
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo dataset_ctc_vvc.csv!" << std::endl;
        return 1;
    }

    std::string line;
    std::getline(file, line); 
    
    std::map<std::string, int> col_to_index;
    int col_idx = 0;
    size_t start = 0, end = line.find(';');
    while (end != std::string::npos) {
        std::string col_name = line.substr(start, end - start);
        col_name.erase(std::remove(col_name.begin(), col_name.end(), '\r'), col_name.end());
        col_to_index[col_name] = col_idx++;
        start = end + 1;
        end = line.find(';', start);
    }
    std::string last_col = line.substr(start);
    last_col.erase(std::remove(last_col.begin(), last_col.end(), '\r'), last_col.end());
    col_to_index[last_col] = col_idx;

    int idx_target_intra = col_to_index[target_col_intra];
    int idx_target_split = col_to_index[target_col_split];
    int idx_video_name   = col_to_index[video_col_name]; 

    std::vector<int> idx_feat_intra, idx_feat_split;
    for (const auto& f : features_intra) idx_feat_intra.push_back(col_to_index[f]);
    for (const auto& f : features_split) idx_feat_split.push_back(col_to_index[f]);

    int TP_intra = 0, TN_intra = 0, FP_intra = 0, FN_intra = 0;
    int TP_split = 0, TN_split = 0, FP_split = 0, FN_split = 0;
    int total_rows = 0;

    std::map<std::string, VideoStats> stats_per_video;

    std::vector<std::string> row_data;
    row_data.reserve(col_to_index.size()); 

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        row_data.clear();
        start = 0;
        end = line.find(';');
        while (end != std::string::npos) {
            row_data.push_back(line.substr(start, end - start));
            start = end + 1;
            end = line.find(';', start);
        }
        row_data.push_back(line.substr(start));

        if (row_data.size() < col_to_index.size()) continue;

        try {
            std::string current_video = row_data[idx_video_name];

            // 1. AVALIAÇÃO: INTRAKEPT
            int true_intra = std::stoi(row_data[idx_target_intra]);
            std::vector<double> fv_intra(features_intra.size());
            for (size_t i = 0; i < features_intra.size(); ++i) {
                fv_intra[i] = std::stod(row_data[idx_feat_intra[i]]); 
            }
            
            int pred_intra = predict_intrakept(fv_intra);

            if (true_intra == 1 && pred_intra == 1) { TP_intra++; stats_per_video[current_video].TP_intra++; }
            else if (true_intra == 0 && pred_intra == 0) { TN_intra++; stats_per_video[current_video].TN_intra++; }
            else if (true_intra == 0 && pred_intra == 1) { FP_intra++; stats_per_video[current_video].FP_intra++; }
            else if (true_intra == 1 && pred_intra == 0) { FN_intra++; stats_per_video[current_video].FN_intra++; }
            stats_per_video[current_video].total_intra++;

            // 2. AVALIAÇÃO: ISSPLIT
            int true_split = std::stoi(row_data[idx_target_split]);
            std::vector<double> fv_split(features_split.size());
            for (size_t i = 0; i < features_split.size(); ++i) {
                fv_split[i] = std::stod(row_data[idx_feat_split[i]]);
            }

            int pred_split = predict_issplit(fv_split);

            if (true_split == 1 && pred_split == 1) { TP_split++; stats_per_video[current_video].TP_split++; }
            else if (true_split == 0 && pred_split == 0) { TN_split++; stats_per_video[current_video].TN_split++; }
            else if (true_split == 0 && pred_split == 1) { FP_split++; stats_per_video[current_video].FP_split++; }
            else if (true_split == 1 && pred_split == 0) { FN_split++; stats_per_video[current_video].FN_split++; }
            stats_per_video[current_video].total_split++;

            total_rows++;
        } catch (const std::exception& e) {
            continue;
        }
    }

    std::cout << "\nTotal de Blocos Processados: " << total_rows << "\n";

    print_report("AVALIACAO INTRAKEPT", "Nao-Intra", "Intra", TP_intra, TN_intra, FP_intra, FN_intra, total_rows);
    print_report("AVALIACAO ISSPLIT", "Nao-Split", "Split", TP_split, TN_split, FP_split, FN_split, total_rows);


    // ==========================================
    // EXPORTAR 1: RESULTADOS POR VÍDEO
    // ==========================================
    std::ofstream out_video_csv("resultados_por_video.csv");
    if (out_video_csv.is_open()) {
        out_video_csv << "VideoName;Classe_VVC;Acuracia_IntraKept_%;Acuracia_IsSplit_%\n";
        
        for (const auto& pair : stats_per_video) {
            const std::string& video_name = pair.first;
            const VideoStats& stats = pair.second;
            
            double acc_intra = stats.total_intra > 0 ? (double)(stats.TP_intra + stats.TN_intra) / stats.total_intra * 100.0 : 0.0;
            double acc_split = stats.total_split > 0 ? (double)(stats.TP_split + stats.TN_split) / stats.total_split * 100.0 : 0.0;
            
            std::string vvc_class = getVideoClass(video_name);

            out_video_csv << std::fixed << std::setprecision(4);
            out_video_csv << video_name << ";" << vvc_class << ";" << acc_intra << ";" << acc_split << "\n";
        }
        out_video_csv.close();
        std::cout << ">>> Arquivo 'resultados_por_video.csv' gerado com sucesso!\n";
    }

    // ==========================================
    // EXPORTAR 2: RESULTADOS AGREGADOS POR CLASSE VVC
    // ==========================================
    std::map<std::string, VideoStats> stats_per_class;
    
    // Agrupa os dados dos vídeos em suas respectivas classes
    for (const auto& pair : stats_per_video) {
        std::string vvc_class = getVideoClass(pair.first);
        stats_per_class[vvc_class].TP_intra += pair.second.TP_intra;
        stats_per_class[vvc_class].TN_intra += pair.second.TN_intra;
        stats_per_class[vvc_class].FP_intra += pair.second.FP_intra;
        stats_per_class[vvc_class].FN_intra += pair.second.FN_intra;
        stats_per_class[vvc_class].total_intra += pair.second.total_intra;

        stats_per_class[vvc_class].TP_split += pair.second.TP_split;
        stats_per_class[vvc_class].TN_split += pair.second.TN_split;
        stats_per_class[vvc_class].FP_split += pair.second.FP_split;
        stats_per_class[vvc_class].FN_split += pair.second.FN_split;
        stats_per_class[vvc_class].total_split += pair.second.total_split;
    }

    std::ofstream out_class_csv("resultados_por_classe.csv");
    if (out_class_csv.is_open()) {
        out_class_csv << "Classe_VVC;Acuracia_IntraKept_%;Acuracia_IsSplit_%\n";
        
        for (const auto& pair : stats_per_class) {
            const std::string& vvc_class = pair.first;
            const VideoStats& stats = pair.second;
            
            double acc_intra = stats.total_intra > 0 ? (double)(stats.TP_intra + stats.TN_intra) / stats.total_intra * 100.0 : 0.0;
            double acc_split = stats.total_split > 0 ? (double)(stats.TP_split + stats.TN_split) / stats.total_split * 100.0 : 0.0;
            
            out_class_csv << std::fixed << std::setprecision(4);
            out_class_csv << vvc_class << ";" << acc_intra << ";" << acc_split << "\n";
        }
        out_class_csv.close();
        std::cout << ">>> Arquivo 'resultados_por_classe.csv' gerado com sucesso!\n" << std::endl;
    }

    return 0;
}