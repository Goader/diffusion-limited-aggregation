set(SOURCE
    ${SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/random_engine.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
    PARENT_SCOPE
)

set(HEADERS
    ${HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation_config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/random_engine.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
    PARENT_SCOPE
)
