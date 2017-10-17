# /mnt/rd is a (mounted) ram disk
THIS_FILE := $(lastword $(MAKEFILE_LIST))
INC_PATH   ?= /mnt/rd/include
CUDA_RD_PATH    ?= /mnt/rd/cuda
#CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_RD_PATH)/include
NVML_INC_PATH ?=/usr/include/nvidia/gdk
NVML_LIB_PATH ?=/usr/src/gdk/nvml/lib
CUDA_COMMON_INC_PATH   ?= $(CUDA_RD_PATH)/src/common/inc
CUDA_LIB_PATH   ?= $(CUDA_RD_PATH)/lib
CUDA_BIN_PATH   ?= $(CUDA_RD_PATH)/bin
OCTAVE_INC_PATH=
OCTAVE_INC_PATH=$(shell test -d /usr/include/octave && echo /usr/include/octave) 
DNAS_PROCESSED=0

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
GENCODE_SM60   	:= -gencode arch=compute_60,code=sm_60 
GENCODE_SM61   	:= -gencode arch=compute_61,code=sm_61 

# keep it lean by only compiling/linking for the gpus you have
ARCHS = $(shell ./tools/gpuArchs)
ifneq ($(filter 3.0,$(ARCHS)),)
    GENCODE_FLAGS += $(GENCODE_SM30)
endif
ifneq ($(filter 3.5,$(ARCHS)),)
    GENCODE_FLAGS += $(GENCODE_SM35)
endif
ifneq ($(filter 5.0,$(ARCHS)),)
    @echo This is some text $(OCTAVE_INC_PATH)
    GENCODE_FLAGS += $(GENCODE_SM50)
endif
ifneq ($(filter 5.2,$(ARCHS)),)
    GENCODE_FLAGS += $(GENCODE_SM52)
endif
ifneq ($(filter 6.1,$(ARCHS)),)
    GENCODE_FLAGS += $(GENCODE_SM61)
    #echo  $(GENCODE_SM61)
endif

#GENCODE_FLAGS :=  $(GENCODE_SM61)$(GENCODE_SM50)

CPPFLAGS= -Wall -m64 -Wno-switch -fopt-info-vec-all -fpic -Wstrict-aliasing=0 -Wno-unused-function -std=c++11 -fopenmp 
AGXXFLAGS= -Wunused-function
NVCPPFLAGS := -Xcompiler -D_FORCE_INLINES -D__CORRECT_ISO_CPP11_MATH_H_PROTO -m64 -Xcompiler '-fpic' -dc -std=c++11  --expt-relaxed-constexpr
LDFLAGS=-lgomp  -lnvidia-ml -L$(CUDA_RD_PATH)/lib64 -L$(NVML_LIB_PATH) 
SRCDIR := src
OGLSRCDIR := src/ogl
TESTDIR := src/test

CUDA_MEMCHECK_FILES:=cuda-memcheck.file*
CUDA_KEEP_FILES=*.cpp?.i? *.cudafe?.* *.cuobjdump *.fatbin?? *.hash *.module_id *.ptx *.cubin

DBG_NAME := cumatrest
EXISTS_DBG_TEST_NAME :=$(shell test -f cumatrest && echo exists) 
RLS_NAME := cumatrel
EXISTS_RLS_TEST_NAME :=$(shell test -f cumatrel && echo exists) 

CONSOLE_OUTPUT := res.txt
SIDEXPS := sidexps
DMG_TOOL := dmg
SAMPLE_DATA_FILES := ex*data*.txt ex4weights.txt ct*.txt train*ubyte

DEBUG := debug
RELEASE := release

FILES_TO_CLEAN = $(DEBUG) $(RELEASE) $(RLS_NAME) $(DBG_NAME) $(CUDA_MEMCHECK_FILES) $(CUDA_KEEP_FILES) a.out

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

# flag to use CuBLAS for some routines (eg matrix product)
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
#ifeq ($(ngen),1)
#	RIBOSOME = echo
#endif

# compile functor kernels templated by functor type polymorphism with static methods (ie function pointers)
ifeq ($(kts),1)
	export cuKts = true
	NVCPPFLAGS += -DCuMatrix_Enable_KTS 
	CPPFLAGS += -DCuMatrix_Enable_KTS 
endif

#debug
ifeq ($(dbg),1)
	OUT_DIR := $(DEBUG)
    CPPFLAGS +=  -g3 -g -rdynamic -ggdb -DCuMatrix_DebugBuild
    #  -lineinfo conflicts with device debug in cuda9
    NVCPPFLAGS += -O0 -g -G -odir $(OUT_DIR) -DCuMatrix_DebugBuild
    LDFLAGS += -g
    OBJDIR := $(DEBUG)
    BASE_NAME := $(DBG_NAME)
    EXISTS_TEST_NAME := $(EXISTS_DBG_TEST_NAME)
else
	OUT_DIR := $(RELEASE)
    NVCPPFLAGS += -odir $(OUT_DIR) 
    CPPFLAGS += -O2 
	OBJDIR := $(RELEASE)
    BASE_NAME := $(RLS_NAME)
    EXISTS_TEST_NAME := $(EXISTS_RLS_TEST_NAME)
endif

# output filenames
LIB_NAME := $(OUT_DIR)/$(BASE_NAME).a
SO_NAME := $(OUT_DIR)/lib$(BASE_NAME).so.1.0.1
TEST_NAME := $(BASE_NAME)
# holds date of .so file (used in test to determine whether executable needs rebuilding)
BEFORE_DATE := `stat -c %y $(SO_NAME)`


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


# enable OMP
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
GEN_CU_SOURCES =$(wildcard $(SRCDIR)/*_Gen.cu) 
GEN_HEADERS =$(wildcard $(SRCDIR)/*_Gen.h) 
#BASE_HEADERS := $(wildcard $(SRCDIR)/*.h)
BUILD_SOURCES= $(BASE_SOURCES) $(BASE_CU_SOURCES) $(BASE_OGL_SOURCES) 

TEST_RUNNERS = testRunner.cc suiteRunner.cc 
TEST_SOURCES= $(filter-out  $(addprefix $(TESTDIR)/, $(TEST_RUNNERS)), \
							 $(wildcard $(TESTDIR)/*.cc))
							 
ifeq ($(ogl),0)
	TEST_SOURCES= $(filter-out *ogl*, $(TEST_SOURCES))
endif
							 
TEST_CU_SOURCES= $(filter-out  $(addprefix $(TESTDIR)/, $(TEST_RUNNERS)), \
							 $(wildcard $(TESTDIR)/*.cu))
#TEST_HEADERS= $(filter-out  $(addprefix $(TESTDIR)/, $(TEST_RUNNERS)), $(wildcard $(TESTDIR)/*.h))
UNIT_TEST_SOURCES=  $(TEST_SOURCES) $(TEST_CU_SOURCES) $(addprefix $(TESTDIR)/,testRunner.cc)
SUITE_TEST_SOURCES= $(TEST_SOURCES) $(TEST_CU_SOURCES) $(addprefix $(TESTDIR)/,suiteRunner.cc)

ifeq ($(suite),1)
	EXEC_TEST_SOURCES= $(SUITE_TEST_SOURCES)
else
	EXEC_TEST_SOURCES= $(UNIT_TEST_SOURCES) 
endif  


# to print the contents of any makefile variable, make echoVar-<<varname>>
echoVar-%  : ; @echo $($*)

HEADERS := $(BUILD_SOURCES:.cc=.h) $(BUILD_SOURCES:.cu=.h)
TEST_HEADERS := $(EXEC_TEST_SOURCES:.cc=.h) $(EXEC_TEST_SOURCES:.cu=.h)  
  
ASPECTS := $(wildcard $(SRCDIR)/**.ah)

# is there a way to do this in one step?
OBJECTS_CC_PASS=$(addprefix $(OBJDIR)/, $(notdir $(BUILD_SOURCES:.cc=.o)))
OBJECTS=$(OBJECTS_CC_PASS:.cu=.o)

TEST_OBJECTS_CC_PASS=$(addprefix $(OBJDIR)/, $(notdir $(EXEC_TEST_SOURCES:.cc=.o)))
TEST_OBJECTS=$(TEST_OBJECTS_CC_PASS:.cu=.o)

# Common includes and paths for CUDA
#-I$(OCTAVE_364_INC_PATH) -I$(OCTAVE_382_INC_PATH) -I$(OCTAVE_401_INC_PATH)
INCLUDES      := -I. -I.. -I$(CUDA_RD_PATH)/samples/common/inc -I$(CUDA_INC_PATH) -I$(NVML_INC_PATH) -I$(OCTAVE_INC_PATH) -I$(CUDA_COMMON_INC_PATH) -I$(INC_PATH) 

all: $(BUILD_SOURCES) $(LIB_NAME)
	
-include $(OBJECTS:.o=.d)


.PHONY : gen_sources
gen_sources:
	@echo $@  # print target name
	@echo GENCODE_FLAGS $(GENCODE_FLAGS)
ifeq (,$(wildcard ./src/*_Gen.*)) 
	$(RIBOSOME) $(DNA_SOURCES)  # generated files don't exist
else	
	@$(MAKE) -f $(THIS_FILE) dna_sources  # generated files exist; test age
endif		
	
.PHONY : dna_sources 
dna_sources: src/BaseUnaryOpIndexF_Gen.h

src/BaseUnaryOpIndexF_Gen.h :  $(DNA_SOURCES)  # arbitrarily test one generated file against the dna file(s) 
	$(RIBOSOME) $(DNA_SOURCES)

# if first execution, first generate functor sources using ribosome, then call make again to ensure that the generated (unless ngen=1 is specified) 
# source files become part of the build 
ifeq ($(MAKELEVEL),0) 
$(LIB_NAME): gen_sources
	@$(MAKE) -s $@
else	
$(LIB_NAME): $(OBJECTS) | $(HEADERS)
ifeq ($(cdp),1)
	$(NVCC) -ccbin g++ -lib -o $(LIB_NAME) $(OBJECTS)
	$(GCC) -shared -o $(SO_NAME) $(OBJECTS)
else
	$(GCC) -shared -o $(SO_NAME)  $(OBJECTS)
	@echo cdp off EXISTS_TEST_NAME: $(EXISTS_TEST_NAME) 
endif
	@echo built:  $(SO_NAME)
ifneq ( $(BEFORE_DATE), `stat -c %y $(SO_NAME)` ) 
ifeq ($(EXISTS_TEST_NAME),exists )
	rm $(TEST_NAME)
	@echo Removed stale test executable $(TEST_NAME)
endif
endif
endif

test: $(TEST_NAME) $(SO_NAME)
ifeq ($(EXISTS_TEST_NAME),exists )
	@printf "\nreplaced: $(TEST_NAME)"
else
	@printf "\nbuilt: $(TEST_NAME)"
endif

$(TEST_NAME): $(TEST_OBJECTS) | $(TEST_HEADERS) 
ifeq ($(cdp),1)
	@echo cdptestname: $(TEST_NAME)
	$(NVCC) -ccbin g++ $(GENCODE_FLAGS) $(TEST_OBJECTS) $(LIB_NAME) $(LDFLAGS) -o $@
else
	@echo basename: $(BASE_NAME)
	$(NVCC) -ccbin g++ $(GENCODE_FLAGS) $(TEST_OBJECTS) $(LIB_NAME) $(LDFLAGS) -o $@
endif
	
VPATH = $(SRCDIR):$(OGLSRCDIR):$(TESTDIR)

# supposed to compile dependent sources when header files change (only works for .cc sources however)
# 'touchImporters.sh' effects this for both architectures (.cu and .cc) using an embedded scala script ich
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
$(TEST_OBJECTS) : | $(OBJDIR)

$(OBJDIR):
	mkdir $(OBJDIR)

#.PHONY : $(HEADERS)
$(HEADERS):
$(TEST_HEADERS):
	
.PHONY : clean
clean:
	rm -rf $(FILES_TO_CLEAN) $(SRCDIR)/*_Gen.*
.PHONY : prepzip

prepzip: clean
	rm -rf $(CONSOLE_OUTPUT) $(DMG_TOOL) $(SAMPLE_DATA_FILES) 
	/bin/sh ./sidexps/clean.sh
