add_library(LLVMCudaFE MODULE mergeKernel.cpp)
set(LLVM_LINK_COMPONENTS Demangle)
llvm_map_components_to_libnames(MY_PASS_LINK_COMPONENTS Demangle)

# this is very important, to link with the passes library
target_link_libraries(LLVMCudaFE ${MY_PASS_LINK_COMPONENTS})
