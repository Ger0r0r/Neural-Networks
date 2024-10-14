cmake_minimum_required(VERSION 3.13)

set(PROJECT_NAME NN_WORK)

project(${PROJECT_NAME} VERSION 1.0 LANGUAGES CXX C)

# Добавляем статические опции линковки
add_link_options(-static -static-libgcc -static-libstdc++)

# Указываем директорию для заголовочных файлов
include_directories(headers)

# Ищем OpenMP
find_package(OpenMP REQUIRED)

# Собираем все cpp-файлы
file(GLOB CPPS "src/*.cpp")

# Устанавливаем C++ стандарт
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Добавляем исполняемый файл
add_executable(${PROJECT_NAME} ${CPPS})

# Добавляем флаги компиляции для поддержки AVX и OpenMP
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -O2 -mavx)

# Линкуем OpenMP библиотеку
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

# Линкуем стандартную библиотеку C
target_link_libraries(${PROJECT_NAME} PRIVATE m)

# Выводим информацию о флагах компиляции для отладки
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
