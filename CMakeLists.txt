cmake_minimum_required(VERSION 3.28)

project(CV2C)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "/usr/local" ${CMAKE_PREFIX_PATH})
# Rechercher tous les fichiers sources .cpp
file(GLOB SOURCES "*.cpp")

# Utiliser find_package pour trouver OpenCV
find_package(OpenCV REQUIRED)

# Utiliser find_package pour trouver Python (remplace FindPythonInterp et FindPythonLibs)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Utiliser find_package pour trouver curl
find_package(CURL REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Freetype REQUIRED)
find_package(HarfBuzz REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(TESSERACT REQUIRED)
include_directories(${Tesseract_INCLUDE_DIR})
link_directories(${TESSERACT_LIBRARY_DIRS})

add_definitions(${TESSERACT_CFLAGS_OTHER})
# Ajouter les exécutables
add_executable(cv2c "${SOURCES}")

# Lier les bibliothèques d'OpenCV, curl et Python
target_link_libraries(cv2c
	CURL::libcurl 
	Python::Python
	${TESSERACT_LIBRARIES}
	opencv_core opencv_imgproc opencv_highgui opencv_videoio opencv_ximgproc
	freetype harfbuzz opencv_freetype
	-lleptonica -ltesseract
)
include_directories(${Tesseract_INCLUDE_DIRS})
