#include "onnx_evaluator.hpp"
#include "softmax.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
// Tenseur d'entree : 119 plans de 8x8.
constexpr int PLANS = 119;
constexpr int COTE = 8;
constexpr std::size_t TAILLE_TENSEUR = PLANS * COTE * COTE;

// Nombre d'elements d'un tenseur de sortie, sans supposer le rang.
std::int64_t nombre_elements(const Ort::Value& tensor) {
    const std::vector<std::int64_t> shape =
        tensor.GetTensorTypeAndShapeInfo().GetShape();
    std::int64_t elements = 1;
    for (const std::int64_t dimension : shape) {
        elements *= dimension;
    }
    return elements;
}
}  // namespace

ONNXEvaluator::ONNXEvaluator(const std::string& model_path, bool use_gpu)
    : env(ORT_LOGGING_LEVEL_ERROR, "AlphaZeroMCTS")
{
    if (use_gpu) {
        // Activation de la carte graphique (Nvidia CUDA)
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0; // Utilise le GPU 0
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    // Optimisations CPU classiques
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::wstring w_model_path(model_path.begin(), model_path.end());
    session = std::make_unique<Ort::Session>(env, w_model_path.c_str(), session_options);
}

void ONNXEvaluator::evaluate_batch(
    const std::vector<float>& input_tensor, 
    std::vector<float>& policies, 
    std::vector<float>& values, 
    int batch_size) {
    evaluate_impl(input_tensor, policies, values, batch_size, true);
}

void ONNXEvaluator::predict_batch(
    const std::vector<float>& input_tensor,
    std::vector<float>& logits,
    std::vector<float>& values,
    int batch_size) {
    evaluate_impl(input_tensor, logits, values, batch_size, false);
}

void ONNXEvaluator::evaluate_impl(
    const std::vector<float>& input_tensor,
    std::vector<float>& policy,
    std::vector<float>& values,
    int batch_size,
    bool probabilities) {

    if (batch_size <= 0) {
        throw std::invalid_argument("evaluateur : taille de lot invalide");
    }
    if (input_tensor.size()
        != static_cast<std::size_t>(batch_size) * TAILLE_TENSEUR) {
        throw std::invalid_argument("evaluateur : taille d'entree invalide");
    }

    std::array<int64_t, 4> input_shape = { batch_size, PLANS, COTE, COTE };
    const char* input_names[] = { "input" };
    const char* output_names[] = { "policy", "value" };

    // Comptes toujours tenus : ils ne dependent pas des horloges.
    m_totals.run_calls++;
    m_totals.evaluated_rows += static_cast<std::uint64_t>(batch_size);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(input_tensor.data()),
        input_tensor.size(), input_shape.data(), input_shape.size()
    );

    std::vector<Ort::Value> output_tensors;
    const auto run_start = m_timing_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    try {
        output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_names, &input_ort, 1, output_names, 2);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "\n[ERREUR FATALE ONNX RUNTIME] : " << e.what() << std::endl;
        throw std::runtime_error(e.what());
    }
    if (m_timing_enabled) {
        m_last_timing.run_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - run_start).count());
        m_totals.timing.run_ns += m_last_timing.run_ns;
    }

    // Les sorties sont controlees AVANT tout acces par indice : un modele
    // incompatible doit etre refuse, pas lu hors bornes.
    if (output_tensors.size() != 2
        || !output_tensors[0].IsTensor() || !output_tensors[1].IsTensor()) {
        throw std::runtime_error(
            "evaluateur : le modele ne rend pas les sorties policy et value");
    }
    if (nombre_elements(output_tensors[0])
            != static_cast<std::int64_t>(batch_size) * POLICY_SIZE
        || nombre_elements(output_tensors[1])
            != static_cast<std::int64_t>(batch_size)) {
        throw std::runtime_error(
            "evaluateur : dimensions de sortie incompatibles");
    }

    const float* policy_data = output_tensors[0].GetTensorData<float>();
    const float* value_data = output_tensors[1].GetTensorData<float>();

    policy.resize(static_cast<std::size_t>(batch_size) * POLICY_SIZE);
    values.assign(value_data, value_data + batch_size);

    const auto post_start = m_timing_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};

    // Softmax indépendant pour CHAQUE position du batch. Le mode parallèle
    // reste disponible dans softmax.hpp, mais il est désactivé ici : en
    // contexte self-play chargé, sa région OpenMP par appel coûte plus cher
    // que le calcul séquentiel (mesure du 2026-09-25).
    if (probabilities) {
        softmax_rows(policy_data, policy.data(), batch_size, POLICY_SIZE, false);
    }
    else {
        std::copy(policy_data, policy_data + policy.size(), policy.data());
    }

    if (m_timing_enabled) {
        m_last_timing.softmax_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - post_start).count());
        m_totals.timing.softmax_ns += m_last_timing.softmax_ns;
    }
}
