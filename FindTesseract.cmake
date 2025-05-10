# FindTesseract.cmake

find_path(Tesseract_INCLUDE_DIR
  NAMES baseapi.h
  HINTS ${CMAKE_INSTALL_PREFIX}/include
  PATH_SUFFIXES tesseract
)

find_library(Tesseract_LIBRARY
  NAMES tesseract
  HINTS ${CMAKE_INSTALL_PREFIX}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Tesseract DEFAULT_MSG Tesseract_LIBRARY Tesseract_INCLUDE_DIR)

mark_as_advanced(Tesseract_INCLUDE_DIR Tesseract_LIBRARY)

set(Tesseract_LIBRARIES ${Tesseract_LIBRARY})
set(Tesseract_INCLUDE_DIRS ${Tesseract_INCLUDE_DIR})

