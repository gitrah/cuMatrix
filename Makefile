# /mnt/rd is a (mounted) ram disk
INC_PATH   ?= /mnt/rd/include
CUDA_RD_PATH    ?= /mnt/rd/cuda
#CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_RD_PATH)/include
NVML_INC_PATH ?=/usr/include/nvidia/gdk
NVML_LIB_PATH ?=/usr/src/gdk/nvml/lib
CUDA_COMMON_INC_PATH   ?= $(CUDA_RD_PATH)/src/common/inc
CUDA_LIB_PATH   ?= $(CUDA_RD_PATH)/lib
CUDA_BIN_PATH   ?= $(CUDA_RD_PATH)/bin
OCTAVE_364_INC_PATH ?= /usr/include/octave-3.6.4
OCTAVE_382_INC_PATH ?= /usr/include/octave-3.8.2

#FREETYPE2_INC_PATH  ?= /usr/include/freetype2
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++
CXX				?= g++
ACXX			?= ag++
# ribosome is a ruby script used for code generation
RIBOSOME		?= ribosome
MAKE			?= /mnt/rd/bin/make

GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35 
GENCODE_SM50   	:= -gencode arch=compute_50,code=sm_50 
GENCODE_SM52   	:= -gencode arch=compute_52,code=sm_52 

# keep it lean by only compiling/linking for the gpus you have
ARCHS = $(shell ./tools/gpuArchs)
ifneq ($(filter 3.0,$(ARCHS)),)
    GENCODE_FLAGS += $(GENCODE_SM30)
endif
ifneq ($(filter 3.5,$(ARCHS)),)
    GENCODE_FLAGS += $(GENCODE_SM35)
endif
ifneq ($(filter 5.0,$(ARCHS)),)
    GENCODE_FLAGS += $(GENCODE_SM50)
endif
ifneq ($(filter 5.2,$(ARCHS)),)
    GENCODE_FLAGS += $(GENCODE_SM52)
endif

CPPFLAGS= -Wall -m64 -Wno-switch -fpic -Wstrict-aliasing=0 -Wno-unused-function -std=c++11 -fopenmp 
AGXXFLAGS= -Wunused-function
# -DCuMatrix_Maxwell_Workaround1
NVCPPFLAGS := -m64 -Xcompiler '-fpic' -dc -std=c++11  --expt-relaxed-constexpr
SRCDIR := src
OGLSRCDIR := src/ogl
TESTDIR := src/test
CUDA_MEMCHECK_FILES:=cuda-memcheck.file*
CUDA_KEEP_FILES=*.cpp?.i? *.cudafe?.* *.cuobjdump *.fatbin?? *.hash *.module_id *.ptx *.cubin

LDFLAGS=-lgomp  -lnvidia-ml -L$(CUDA_RD_PATH)/lib64 -L$(NVML_LIB_PATH) 
DBG_EXECUTABLE := cumatrest
RLS_EXECUTABLE := cumatrel
CONSOLE_OUTPUT := res.txt
SIDEXPS := sidexps
DMG_TOOL := dmg
SAMPLE_DATA_FILES := ex*data*.txt ex4weights.txt ct*.txt

DEBUG := debug
RELEASE := release

FILES_TO_CLEAN = $(DEBUG) $(RELEASE) $(RLS_EXECUTABLE) $(DBG_EXECUTABLE) $(CUDA_MEMCHECK_FILES) $(CUDA_KEEP_FILES) a.out

# keep (kep) retains ptx files for seeing things like kernel register counts
ifeq ($(kep),1)
	NVCPPFLAGS += -keep
endif
ifeq ($(rpo),1)
	CPPFLAGS += -frepo
endif
  
# verbose nvcc compilation
ifeq ($(vrb),1)
	NVCPPFLAGS +=  -Xptxas="-v"
endif

#nvml for gpu temp
ifeq ($(nvml),1)
    CPPFLAGS +=  -DCuMatrix_NVML
	NVCPPFLAGS += -DCuMatrix_NVML
endif

ifeq ($(blas),1)
    CPPFLAGS +=  -DCuMatrix_UseCublas
	NVCPPFLAGS += -DCuMatrix_UseCublas
	LDFLAGS +=  -lcublas  
endif

# compile functor kernels parameterized by functor data type and functor state dimension
# have CuFunctor.dna simulate functor polymorphism with static methods (ie function pointers)
ifeq ($(statFunc),1)
    CPPFLAGS +=  -DCuMatrix_StatFunc
	NVCPPFLAGS += -DCuMatrix_StatFunc
	export cufuncStatic = true
endif

#disable ribosome so makefile generates only binaries
ifeq ($(ngen),1)
	RIBOSOME = echo
endif

# compile functor kernels templated by functor type polymorphism with static methods (ie function pointers)
ifeq ($(kts),1)
	export cuKts = true
	NVCPPFLAGS += -DCuMatrix_Enable_KTS 
	CPPFLAGS += -DCuMatrix_Enable_KTS 
endif

#debug
ifeq ($(dbg),1)
	OUT_DIR := $(DEBUG)
    CPPFLAGS +=  -g3 -g -rdynamic -ggdb
    NVCPPFLAGS += -O0 -g -G -lineinfo -odir $(OUT_DIR) -DCuMatrix_DebugBuild
    LDFLAGS += -g
    OBJDIR := $(DEBUG)
    EXECUTABLE := $(DBG_EXECUTABLE)
else
	OUT_DIR := $(RELEASE)
    NVCPPFLAGS += -odir $(OUT_DIR) 
    CPPFLAGS += -O2 
	OBJDIR := $(RELEASE)
    EXECUTABLE := $(RLS_EXECUTABLE)
endif


# build version using cuda dynamic parallelism 
ifeq ($(cdp),1)
	LDFLAGS += -lcudadevrt -lcudart
	NVCPPFLAGS += -DCuMatrix_Enable_Cdp 
else
	LDFLAGS += -lcudart -lcublas 
endif

# opengl
ifeq ($(ogl),1)
	NVCPPFLAGS += -DCuMatrix_Enable_Ogl
	BASE_OGL_SOURCES := $(wildcard $(OGLSRCDIR)/*.cc)
	LDFLAGS += -lGL -lglut -lGLU
else
	BASE_OGL_SOURCES =
	
endif

# don't optimize
ifeq ($(nopt),1)
    CPPFLAGS += -O0 
    NVCPPFLAGS += -O0
endif


# omp
ifeq ($(omp),1)
    CPPFLAGS += -DCuMatrix_UseOmp 
    NVCPPFLAGS += -DCuMatrix_UseOmp 
endif


# keep i86 assembly, cuda intermediate sources
ifeq ($(assy),1)
    CPPFLAGS += -S 
	NVCPPFLAGS += -keep
endif

#all cpp source
BASE_SOURCES := $(wildcard $(SRCDIR)/*.cc)
DNA_SOURCES := $(wildcard $(SRCDIR)/*.dna)
BASE_CU_SOURCES = $(wildcard $(SRCDIR)/*.cu)
#BASE_HEADERS := $(wildcard $(SRCDIR)/*.h)
TEST_RUNNERS = testRunner.cc suiteRunner.cc 
TEST_SOURCES= $(filter-out  $(addprefix $(TESTDIR)/, $(TEST_RUNNERS)), \
							 $(wildcard $(TESTDIR)/*.cc))
TEST_CU_SOURCES= $(filter-out  $(addprefix $(TESTDIR)/, $(TEST_RUNNERS)), \
							 $(wildcard $(TESTDIR)/*.cu))
#TEST_HEADERS= $(filter-out  $(addprefix $(TESTDIR)/, $(TEST_RUNNERS)), $(wildcard $(TESTDIR)/*.h))
UNIT_TEST_SOURCES= $(BASE_SOURCES) $(BASE_CU_SOURCES) $(BASE_OGL_SOURCES) $(TEST_SOURCES) $(TEST_CU_SOURCES) $(addprefix $(TESTDIR)/,testRunner.cc)
SUITE_TEST_SOURCES= $(BASE_SOURCES) $(BASE_CU_SOURCES) $(TEST_SOURCES) $(TEST_CU_SOURCES) $(addprefix $(TESTDIR)/,suiteRunner.cc)

ifeq ($(suite),1)
	BUILD_SOURCES= $(SUITE_TEST_SOURCES)
else
	BUILD_SOURCES= $(UNIT_TEST_SOURCES)
endif  

ifeq ($(ogl),0)
	TEST_SOURCES= $(filter-out *ogl*, $(TEST_SOURCES))
endif

# to print the contents of any makefile variable, make echoVar-<<varname>>
echoVar-%  : ; @echo $($*)

HEADERS := $(BUILD_SOURCES:.cc=.h) $(BUILD_SOURCES:.cu=.h)  
ASPECTS := $(wildcard $(SRCDIR)/**.ah)

# is there a way to do this in one step?
OBJECTS_CC_PASS=$(addprefix $(OBJDIR)/, $(notdir $(BUILD_SOURCES:.cc=.o)))
OBJECTS=$(OBJECTS_CC_PASS:.cu=.o)

# Common includes and paths for CUDA
INCLUDES      := -I. -I.. -I$(CUDA_RD_PATH)/samples/common/inc -I$(CUDA_INC_PATH) -I$(NVML_INC_PATH) -I$(OCTAVE_364_INC_PATH) -I$(OCTAVE_382_INC_PATH) -I$(CUDA_COMMON_INC_PATH) -I$(INC_PATH) 

all: $(BUILD_SOURCES) $(EXECUTABLE)
	
-include $(OBJECTS:.o=.d)

.PHONY : gen_sources
gen_sources : 
	$(RIBOSOME) $(DNA_SOURCES)
	
# if first execution, first generate sources, then call make again to ensure that the generated (unless ngen=1 is specified) 
# source files become part of the build 
ifeq ($(MAKELEVEL),0) 
$(EXECUTABLE): gen_sources
	@$(MAKE) -s $@
else	
$(EXECUTABLE): $(OBJECTS) | $(HEADERS)
ifeq ($(cdp),1)
	$(NVCC) -g $(GENCODE_FLAGS) -rdc=true -dlink $(OBJECTS) -lcudadevrt -o $(OUT_DIR)/link.o
	$(GCC) -rdynamic $(OBJECTS) $(OUT_DIR)/link.o $(LDFLAGS) -o $@
else
	$(NVCC) $(GENCODE_FLAGS) -Xcompiler '-fPIC' -dlink $(OBJECTS) -o $(OUT_DIR)/link.o
	$(GCC) $(OBJECTS) $(OUT_DIR)/link.o $(LDFLAGS) -lcudart -o $@
endif
endif

test: $(EXECUTABLE)
	$(EXECUTABLE)
	
VPATH = $(SRCDIR):$(OGLSRCDIR):$(TESTDIR)

# supposed to compile dependent sources when header files change (for .cc sources anyway)
# 'touchImporters.sh' effects forces this for both architectures 
$(OBJDIR)/%.o : %.cc
	$(GCC) -c $(CPPFLAGS) $(INCLUDES) $^ -o $(OBJDIR)/$*.o
	$(GCC) -MM $(CPPFLAGS) $^ > $(OBJDIR)/$*.d
	@cp -f $(OBJDIR)/$*.d $(OBJDIR)/$*.d.tmp
	@sed -e 's/.*://' -e 's/\\$$//' < $(OBJDIR)/$*.d.tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$/:/' >> $(OBJDIR)/$*.d
	@rm -f $(OBJDIR)/$*.d.tmp


$(OBJDIR)/%.o : %.cu
	$(NVCC) $(NVCPPFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -c $^ -o $(OBJDIR)/$*.o

$(OBJECTS) : | $(OBJDIR)

$(OBJDIR):
	mkdir $(OBJDIR)

#.PHONY : $(HEADERS)
$(HEADERS):
	
.PHONY : clean
clean:
	rm -rf $(FILES_TO_CLEAN) $(SRCDIR)/*_Gen.*
.PHONY : prepzip

prepzip: clean
	rm -rf $(CONSOLE_OUTPUT) $(DMG_TOOL) $(SAMPLE_DATA_FILES) 
	/bin/sh ./sidexps/clean.sh
