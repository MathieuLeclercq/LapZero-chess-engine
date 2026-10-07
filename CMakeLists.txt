cmake_minimum_required(VERSION 3.24)
project(Chess)
enable_testing()
cmake_policy(SET CMP0135 NEW)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# ==========================================
# RÉCUPÉRATION DE ONNX RUNTIME (C++, Windows x64)
# ==========================================
set(ONNXRUNTIME_VERSION "1.24.3")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}.zip")
include(FetchContent)

FetchContent_Declare(
  onnxruntime
  URL "${ONNXRUNTIME_URL}"
  URL_HASH SHA256=f2af33fea71f477027961b8027e7496ab01c6b962ae4f596bf8163192b03b326
)
FetchContent_MakeAvailable(onnxruntime)

# Extraction des chemins vers les dossiers de l'archive téléchargée
FetchContent_GetProperties(onnxruntime SOURCE_DIR onnxruntime_src)
set(ONNX_GPU_DIR "${onnxruntime_src}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNX_GPU_DIR}/include")
set(ONNXRUNTIME_LIB "${ONNX_GPU_DIR}/lib/onnxruntime.lib")
set(ONNXRUNTIME_DLL "${ONNX_GPU_DIR}/lib/onnxruntime.dll")
set(ONNXRUNTIME_SHARED_DLL "${ONNX_GPU_DIR}/lib/onnxruntime_providers_shared.dll")
set(ONNXRUNTIME_CUDA_DLL "${ONNX_GPU_DIR}/lib/onnxruntime_providers_cuda.dll")
# ==========================================
# 1. Création de la librairie core
# ==========================================
add_library(chess_core STATIC 
    src/chessboard.cpp
    src/evaluation_cache.cpp
    src/evaluation_key.cpp
    src/square.cpp 
    src/piece.cpp 
    src/move.cpp 
    src/pgn_parser.cpp
    src/mcts.cpp 
    src/mcts_observe.cpp
    src/mcts_batch.cpp
    src/mcts_reservation.cpp
    src/mcts_wave.cpp
    src/search_executor.cpp
    src/search_timing.cpp
    src/zobrist.cpp
    src/onnx_evaluator.cpp
    src/selfplay_manager.cpp
    src/perft.cpp)

# On indique au compilateur où chercher les headers locaux ET ceux de ONNX Runtime
target_include_directories(chess_core PUBLIC src ${ONNXRUNTIME_INCLUDE_DIR})

# On lie la librairie ONNX Runtime
target_link_libraries(chess_core PUBLIC ${ONNXRUNTIME_LIB} OpenMP::OpenMP_CXX Threads::Threads)

# ==========================================
# 2. Exécutable de test (C++ pur)
# ==========================================
add_executable(chess_tests src/main.cpp)
target_link_libraries(chess_tests PRIVATE chess_core)
add_test(NAME chess_tests_fen_smoke
    COMMAND chess_tests --fen-file ${CMAKE_SOURCE_DIR}/tests/data/fens_smoke.txt)
add_test(NAME chess_tests_pgn_smoke
    COMMAND chess_tests --pgn ${CMAKE_SOURCE_DIR}/tests/data/pgn_smoke.pgn)
set_tests_properties(chess_tests_pgn_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "PGN : 1 parties et 20 demi-coups verifies")

# ==========================================
# 2 bis. Executable de validation perft
# ==========================================
add_executable(chess_perft src/perft_main.cpp)
target_link_libraries(chess_perft PRIVATE chess_core)

# Perft dans CTest : palier rapide instantane, puis palier calibre a
# profondeurs figees (environ 43,5 M de noeuds, 23 s attendues en Release).
add_test(NAME chess_perft_bench
    COMMAND chess_perft bench --strict --check-fen)
add_test(NAME chess_perft_roundtrip
    COMMAND chess_perft roundtrip)
add_test(NAME chess_perft_calibre
    COMMAND chess_perft calibre --strict --check-fen)
set_tests_properties(chess_perft_calibre PROPERTIES TIMEOUT 300)

# ==========================================
# 2 ter. Tests unitaires C++ du MCTS
# ==========================================
function(copy_onnx_runtime target_name)
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:${target_name}>"
        VERBATIM)
endfunction()

add_executable(search_timing_tests tests/cpp/test_search_timing.cpp)
target_link_libraries(search_timing_tests PRIVATE chess_core)
copy_onnx_runtime(search_timing_tests)
add_test(NAME search_timing COMMAND search_timing_tests)

add_executable(evaluator_contract_tests tests/cpp/test_evaluator.cpp)
target_link_libraries(evaluator_contract_tests PRIVATE chess_core)
copy_onnx_runtime(evaluator_contract_tests)
add_test(NAME evaluator_contract COMMAND evaluator_contract_tests)

add_executable(board_copy_tests tests/cpp/test_board_copy.cpp)
target_link_libraries(board_copy_tests PRIVATE chess_core)
copy_onnx_runtime(board_copy_tests)
add_test(NAME board_copy COMMAND board_copy_tests)

add_executable(softmax_rows_tests tests/cpp/test_softmax_rows.cpp)
target_link_libraries(softmax_rows_tests PRIVATE chess_core)
copy_onnx_runtime(softmax_rows_tests)
add_test(NAME softmax_rows COMMAND softmax_rows_tests)

add_executable(evaluation_cache_tests tests/cpp/test_evaluation_cache.cpp)
target_link_libraries(evaluation_cache_tests PRIVATE chess_core)
copy_onnx_runtime(evaluation_cache_tests)
add_test(NAME evaluation_cache COMMAND evaluation_cache_tests)

add_executable(reservation_tests tests/cpp/test_reservation.cpp)
target_link_libraries(reservation_tests PRIVATE chess_core)
copy_onnx_runtime(reservation_tests)
add_test(NAME reservation COMMAND reservation_tests)

add_executable(node_state_tests tests/cpp/test_node_state.cpp)
target_link_libraries(node_state_tests PRIVATE chess_core)
copy_onnx_runtime(node_state_tests)
add_test(NAME node_state COMMAND node_state_tests)

add_executable(search_executor_tests tests/cpp/test_search_executor.cpp)
target_link_libraries(search_executor_tests PRIVATE chess_core Threads::Threads)
copy_onnx_runtime(search_executor_tests)
add_test(NAME search_executor COMMAND search_executor_tests)
set_tests_properties(search_executor PROPERTIES TIMEOUT 30)

add_executable(wave_collection_tests tests/cpp/test_wave_collection.cpp)
target_link_libraries(wave_collection_tests PRIVATE chess_core Threads::Threads)
copy_onnx_runtime(wave_collection_tests)
add_test(NAME wave_collection COMMAND wave_collection_tests)
set_tests_properties(wave_collection PROPERTIES TIMEOUT 30)

add_executable(wave_search_tests tests/cpp/test_wave_search.cpp)
target_link_libraries(wave_search_tests PRIVATE chess_core Threads::Threads)
copy_onnx_runtime(wave_search_tests)
add_test(NAME wave_search COMMAND wave_search_tests)
set_tests_properties(wave_search PROPERTIES TIMEOUT 30)

add_executable(multicore_stress_tests tests/cpp/test_multicore_stress.cpp)
target_link_libraries(multicore_stress_tests PRIVATE chess_core Threads::Threads)
copy_onnx_runtime(multicore_stress_tests)
add_test(NAME multicore_stress COMMAND multicore_stress_tests)
set_tests_properties(multicore_stress PROPERTIES TIMEOUT 180)

add_executable(selfplay_shared_core_tests
    tests/cpp/test_selfplay_shared_core.cpp)
target_link_libraries(selfplay_shared_core_tests PRIVATE chess_core)
copy_onnx_runtime(selfplay_shared_core_tests)
add_test(NAME selfplay_shared_core COMMAND selfplay_shared_core_tests)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/selfplay_test_workdir")
set_tests_properties(selfplay_shared_core PROPERTIES
    TIMEOUT 180
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/selfplay_test_workdir")

add_executable(selfplay_phase_bench tests/cpp/selfplay_phase_bench.cpp)
target_link_libraries(selfplay_phase_bench PRIVATE chess_core)
copy_onnx_runtime(selfplay_phase_bench)
add_custom_command(TARGET selfplay_phase_bench POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${ONNXRUNTIME_SHARED_DLL}" "$<TARGET_FILE_DIR:selfplay_phase_bench>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${ONNXRUNTIME_CUDA_DLL}" "$<TARGET_FILE_DIR:selfplay_phase_bench>"
    VERBATIM)
add_test(NAME selfplay_phase_bench_smoke
    COMMAND selfplay_phase_bench --fake --repetitions 1 --rounds 2)
set_tests_properties(selfplay_phase_bench_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION
        "\\\"nn_calls\\\": 16, \\\"nn_batches\\\": 2, \\\"tt_hits\\\": [0-9]+, \\\"tt_misses\\\": [0-9]+, \\\"tt_position_matches\\\":")

add_executable(evaluator_batch_bench tests/cpp/evaluator_batch_bench.cpp)
target_link_libraries(evaluator_batch_bench PRIVATE chess_core)
copy_onnx_runtime(evaluator_batch_bench)
add_custom_command(TARGET evaluator_batch_bench POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${ONNXRUNTIME_SHARED_DLL}" "$<TARGET_FILE_DIR:evaluator_batch_bench>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${ONNXRUNTIME_CUDA_DLL}" "$<TARGET_FILE_DIR:evaluator_batch_bench>"
    VERBATIM)
add_test(NAME evaluator_batch_bench_smoke
    COMMAND evaluator_batch_bench --fake --batches 1 4 8
            --repetitions 2 --rounds 2)
set_tests_properties(evaluator_batch_bench_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION
        "\\\"batch\\\": 8, \\\"repetition\\\": 1, \\\"calls\\\": 2")

# ==========================================
# 3. Configuration Python & Pybind11
# ==========================================
find_package(Python3 3.13 COMPONENTS Interpreter Development REQUIRED)

# Transmet dynamiquement la version trouvée à Pybind11
set(PYBIND11_PYTHON_VERSION ${Python3_VERSION} CACHE STRING "Version Python pour pybind11" FORCE)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v3.0.3
)
FetchContent_MakeAvailable(pybind11)

# ==========================================
# 4. Création du module Python
# ==========================================
pybind11_add_module(chess_engine src/bindings.cpp)
target_link_libraries(chess_engine PRIVATE chess_core)

# Génération des stubs ET copie de la DLL ONNX
add_custom_command(TARGET chess_engine POST_BUILD
    # Copie de la DLL principale
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${ONNXRUNTIME_DLL}"
        "${CMAKE_SOURCE_DIR}/python_src/onnxruntime.dll"
    
    # Ajout pour le GPU
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${ONNXRUNTIME_SHARED_DLL}"
        "${CMAKE_SOURCE_DIR}/python_src/onnxruntime_providers_shared.dll"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${ONNXRUNTIME_CUDA_DLL}"
        "${CMAKE_SOURCE_DIR}/python_src/onnxruntime_providers_cuda.dll"
        
    # Génération des stubs (.pyi) en utilisant l'exécutable Python trouvé dynamiquement
    COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CMAKE_SOURCE_DIR}/python_src"
            "${Python3_EXECUTABLE}" -m pybind11_stubgen chess_engine
            --enum-class-locations "PieceType:chess_engine"
            --output-dir "${CMAKE_SOURCE_DIR}/python_src"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/python_src"
    COMMENT "Copie de onnxruntime.dll et génération des stubs..."
    VERBATIM
)

# Force CMake à exporter le module compilé dans le dossier python_src/
set_target_properties(chess_engine PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python_src)
set_target_properties(chess_engine PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/python_src)
