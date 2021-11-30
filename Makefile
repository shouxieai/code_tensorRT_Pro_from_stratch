# ---------------- g++ and nvcc compiler ------------------------------------------------------------------
cc   := g++
nvcc  := $(lean_cuda)/bin/nvcc
# difference between := and = .refer to https://stackoverflow.com/questions/4879592/whats-the-difference-between-and-in-makefile
# this part lets you access g++ and nvcc compile

# ----------------- GPU architecture ------------------------------------------------------
# if using other graphic cards, modify -gencode=compute_75,code=sm_75 to corresponding compute capability. 
# more details about GPU Compute Capability can be refered to here. https://developer.nvidia.com/zh-cn/cuda-gpus#compute
cuda_arch := -gencode=compute_75,code=sm_75
# -arch specifies the name of GPU architecture
# todo PTX, -arch, -gencode
# arch and gencode :https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# multiple computexx and codexx can be added here for a more general compatibility.


# ------------------ cpp and o file creation and replacement --------------------------------------------------------------------------
# find all related cpp file
cpp_srcs := $(shell find src -name "*.cpp")
# replace the extension with cpp.o
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:src/%=objs/%)
# substitution reference: https://www.gnu.org/software/make/manual/make.html#:~:text=6.3.1-,Substitution%20References,-A%20substitution%20reference
cpp_mk   := $(cpp_objs.cpp.o=.cpp.mk)

cu_srcs  := $(shell find src -name "*.cu")
cu_objs  := $(cu_srcs:.cu=.cu.o)
cu_objs  := $(cu_objs:src/%=objs/%)
cu_mk    := $(cu_objs:.cu.o=.cu.mk)

# ------------------ lean path, header path, lib path and lib --------------------------------------------------------------------------
lean_protobuf  := /datav/lean/protobuf3.11.4
lean_tensor_rt := /datav/lean/TensorRT-8.0.3.4.cuda11.3.cudnn8.2
lean_cudnn     := /datav/lean/cudnn8.2.2.26
lean_opencv    := /datav/lean/opencv-4.2.0
lean_cuda      := /datav/lean/cuda-11.2
use_python     := true
python_root    := /datav/software/anaconda3
python_name    := python3.8

# the above explained with metaphors and analogies
# the relationship among cuda, cudnn, protobuf and tensorRT
# - ref: https://blog.csdn.net/weixin_42370067/article/details/106135411
#
# - cuda:       computing architecture             
#				---> linking GPU and application. User manipulates GPU by APIs offered by CUDA
#				(seen as cuda offers a series of commands which you use to dictate GPU to get sth done)
# 
# - cuDNN:      accelerating/optimization library 
#			    ---> optimize the model before calling the CUDA APIs
# 
# - TensorRT:   accelerating/optimization library
#               ---> further optimize based on cuDNN. Built for inference mainly.
#               ref: https://forums.developer.nvidia.com/t/tensorrt-vs-cudnn/155931/3
#               imgs: interactions_cuda_cudnn_tensorRT.png
# 
# - protobuf:   Protocol Buffers is a free and open-source cross-platform data format used to 
#               serialize structured data. It is useful in developing programs to communicate with each other over a network or for storing data.
#               e.g. 
# 				An end-to-end sample that trains a model in TensorFlow and pytorch, freezes the model and writes it to a protobuf file, converts it to UFF, and finally runs inference using TensorRT.				
#				ref: https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html#:~:text=An%20end-to-end%20sample%20that%20trains%20a%20model%20in%20TensorFlow%20and%20Keras%2C%20freezes%20the%20model%20and%20writes%20it%20to%20a%20protobuf%20file%2C%20converts%20it%20to%20UFF%2C%20and%20finally%20runs%20inference%20using%20TensorRT.

# the above path as a root to each denpendency for accessing .hpp and .so or .a etc
# note that no space after slash 
include_paths := src                 \
			src/application      \
			src/tensorRT         \
			src/tensorRT/common  \
			$(lean_protobuf)/include \
			$(lean_opencv)/include/opencv4 \
			$(lean_tensor_rt)/include \
			$(lean_cuda)/include  \
			$(lean_cudnn)/include


# note that some might be lib64. Others lib
library_paths := $(lean_protobuf)/lib \
			$(lean_opencv)/lib    \
			$(lean_tensor_rt)/lib \
			$(lean_cuda)/lib64  \
			$(lean_cudnn)/lib

# todo detail them
link_librarys := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs opencv_highgui \
			nvinfer nvinfer_plugin \
			cuda cublas cudart cudnn \
			stdc++ protobuf dl




# ------------------ python support --------------------------------------------------------------------------
# HAS_PYTHON indicates whether supporting python api
support_define :=

# conditional directive  ref: https://www.gnu.org/software/make/manual/html_node/Conditional-Example.html#:~:text=ifeq%20directive%20begins%20the%20conditional%2C%20and
ifeq ($(use_python), true)
include_paths  += $(python_root)/include/$(python_name)
library_paths  += $(python_root)/lib
link_librarys  += $(python_name)
support_define += -DHAS_PYTHON
endif



# ------------------ complete the include, library, run path etc --------------------------------------------------------------------------

empty         :=
export_path   := $(subst $(empty) $(empty),:,$(library_paths))


# linking with [-l] [-L] or [-Wl,-rpath]
#! ref: (highly recommended)https://gcc.gnu.org/legacy-ml/gcc-help/2005-12/msg00017.html 
#
# -L: supplies a path(dir) that is to be searched during only !!LINKING TIME!!. However -L doesn't ensure you working through well till the end. 
# -l: specifies a library(e.g. foo.so or foo.a) to link
# -Wl: tells the run-time loaders where it looks for the shared libraray.
# e.g. gcc -o foo foo.c -L/usr/local/lib -lfoo -Wl,-rpath=/usr/local/lib
#
# ref:
# write a new dynamic library into default search path
# 	- https://note.youdao.com/ynoteshare/index.html?id=681164040148bdb930dd0b5906a14955&type=note&_time=1638173315586

include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))
run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))


# ------------------ compiling and linking flag --------------------------------------------------------------------------
#todo options are to be explained 
cpp_compile_flags := -std=c++11 -g -w -O0 -fPIC -pthread -fopenmp $(support_define)
cu_compile_flags  := -std=c++11 -g -w -O0 -Xcompiler "$(cpp_compile_flags)" $(cuda_arch) $(support_define)
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'
# notice that -Xcompiler "$(xxxx)" means all args of xxxx will be passed in -Xcompiler as a whole

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)

#todo to explain
ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif




# ------------------ make xxx --------------------------------------------------------------------------
pro : workspace/pro
trtpyc: python/trtpy/libtrtpyc.so


workspace/pro: $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)
# $@ all targets     $^ all dependencies    $< 1-st prerequisite

objs/%.cpp.o : src/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)
	
objs/%.cu.o : src/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)


yolo: workspace/pro
	@cd workspace && ./pro yolo

alphapose : workspace/pro
	@cd workspace && ./pro alphapose

fall : workspace/pro
	@cd workspace && ./pro fall_recognize


clean :
	@rm -rf objs


.PHONY : clean yolo alphapose fall debug














