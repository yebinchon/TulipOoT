add_library(OutlineKernel MODULE outlineKernel.cpp)

# this is very important, to link with the passes library
target_link_libraries(OutlineKernel ${llvm_libs})
