# the -include and gcc -MM enable dependency mechanism
# that will correctly reflect changes in template or other .h files
CUDA_RD_PATH    ?= /mnt/rd/cuda
CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_RD_PATH)/include
CUDA_LIB_PATH   ?= $(CUDA_RD_PATH)/lib
CUDA_BIN_PATH   ?= $(CUDA_RD_PATH)/bin
 
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++
CXX				?= g++
ACXX			?= ag++

GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
#GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)
GENCODE_FLAGS   := $(GENCODE_SM30)
# Debug build flags

CPPFLAGS=-c -Wall -m64 -Wno-switch -frepo -g
AGXXFLAGS= -Wunused-function
#-fpermissive
NVCPPFLAGS := -m64
SRCDIR := src
TESTDIR := src/test

LDFLAGS=-L$(CUDA_LIB_PATH)64 -lcudart
DBG_EXECUTABLE := cumatrest
RLS_EXECUTABLE := cumatrel

DEBUG := debug
RELEASE := release
ifeq ($(dbg),1)
     NVCPPFLAGS += -g -G -Xptxas="-v" 
    LDFLAGS += -rdynamic
    OBJDIR := $(DEBUG)
    EXECUTABLE := $(DBG_EXECUTABLE)
else
     CPPFLAGS += -O2 
	OBJDIR := $(RELEASE)
    EXECUTABLE := $(RLS_EXECUTABLE)
endif


#all cpp source
BASE_SOURCES := $(wildcard $(SRCDIR)/*.cc)
#BASE_HEADERS := $(wildcard $(SRCDIR)/*.h)
TEST_RUNNERS = testRunner.cc suiteRunner.cc 
TEST_SOURCES= $(filter-out  $(addprefix $(TESTDIR)/, $(TEST_RUNNERS)), \
							 $(wildcard $(TESTDIR)/*.cc))
#TEST_HEADERS= $(filter-out  $(addprefix $(TESTDIR)/, $(TEST_RUNNERS)), $(wildcard $(TESTDIR)/*.h))
UNIT_TEST_SOURCES= $(BASE_SOURCES) $(TEST_SOURCES) $(addprefix $(TESTDIR)/,testRunner.cc)
SUITE_TEST_SOURCES= $(BASE_SOURCES) $(TEST_SOURCES) $(addprefix $(TESTDIR)/,suiteRunner.cc)


ifeq ($(suite),1)
	BUILD_SOURCES= $(SUITE_TEST_SOURCES)
else
	BUILD_SOURCES= $(UNIT_TEST_SOURCES)
endif  

HEADERS := $(BUILD_SOURCES:.cc=.h)
ASPECTS := $(wildcard $(SRCDIR)/**.ah)

#$(BUILD_SOURCES) : $(HEADERS)

OBJECTS=$(addprefix $(OBJDIR)/, $(notdir $(BUILD_SOURCES:.cc=.o)) \
	CuMatrix.o CuMatrixProduct.o CuMatrixReductions.o CuMatrixSimpleReductions.o CuMatrixIndexedReductions.o CuMatrixForm.o CuMatrixRand.o)  

# Common includes and paths for CUDA
INCLUDES      := -I. -I.. -I$(CUDA_PATH)/samples/common/inc -I$(CUDA_INC_PATH) 

all: $(BUILD_SOURCES) $(EXECUTABLE)
	
-include $(OBJECTS:.o=.d)
	
$(EXECUTABLE): $(OBJECTS) | $(HEADERS)
	$(GCC) $(LDFLAGS) $(OBJECTS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)
	
VPATH = $(SRCDIR):$(TESTDIR)

$(OBJDIR)/%.o : %.cc
	$(GCC) -c $(CPPFLAGS) $(INCLUDES) $^ -o $(OBJDIR)/$*.o
	$(GCC) -MM $(CPPFLAGS) $^ > $(OBJDIR)/$*.d
	@cp -f $(OBJDIR)/$*.d $(OBJDIR)/$*.d.tmp
	@sed -e 's/.*://' -e 's/\\$$//' < $(OBJDIR)/$*.d.tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$/:/' >> $(OBJDIR)/$*.d
	@rm -f $(OBJDIR)/$*.d.tmp

#$(OBJDIR)/%.o : %.cc
#	@echo Compiling $<
#	$(ACXX) $(CPPFLAGS) $(AGXXFLAGS) $(INCLUDES) -x c++ -c $< -o $@

something : else
#	#$(ACXX) -MM $(CPPFLAGS) $(AGXXFLAGS) $^ > $(OBJDIR)/$*.d
	#@cp -f $(OBJDIR)/$*.d $(OBJDIR)/$*.d.tmp
	#@sed -e 's/.*://' -e 's/\\$$//' < $(OBJDIR)/$*.d.tmp | fmt -1 | \
	#  sed -e 's/^ *//' -e 's/$$/:/' >> $(OBJDIR)/$*.d
	#@rm -f $(OBJDIR)/$*.d.tmp


#CuMatrix.cu
$(OBJDIR)/CuMatrix.o:  $(addprefix $(SRCDIR)/, \
	CuMatrix.cu CuMatrix.h)
	$(NVCC) $(NVCPPFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<
#CuMatrixRand.cu
$(OBJDIR)/CuMatrixRand.o:  $(addprefix $(SRCDIR)/, \
	CuMatrixRand.cu CuMatrix.h)
	$(NVCC) $(NVCPPFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<
#CuMatrixProduct.cu
$(OBJDIR)/CuMatrixProduct.o:  $(addprefix $(SRCDIR)/, \
	CuMatrixProduct.cu CuMatrix.h)
	$(NVCC) $(NVCPPFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<
#CuMatrixReductions.cu
$(OBJDIR)/CuMatrixReductions.o:  $(addprefix $(SRCDIR)/, \
	CuMatrixReductions.cu CuMatrix.h)
	$(NVCC) $(NVCPPFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<
#CuMatrixSimpleReductions.cu
$(OBJDIR)/CuMatrixSimpleReductions.o:  $(addprefix $(SRCDIR)/, \
	CuMatrixSimpleReductions.cu CuMatrix.h)
	$(NVCC) $(NVCPPFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<
$(OBJDIR)/CuMatrixIndexedReductions.o:  $(addprefix $(SRCDIR)/, \
	CuMatrixIndexedReductions.cu CuMatrix.h)
	$(NVCC) $(NVCPPFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<
#CuMatrixForm.cu
$(OBJDIR)/CuMatrixForm.o:  $(addprefix $(SRCDIR)/, \
	CuMatrixForm.cu CuMatrix.h)
	$(NVCC) $(NVCPPFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<
	
$(OBJECTS) : | $(OBJDIR)

$(OBJDIR):
	mkdir $(OBJDIR)

#.PHONY : $(HEADERS)
$(HEADERS):
	
.PHONY : clean
clean:
	rm -rf $(DEBUG) $(RELEASE) $(RLS_EXECUTABLE) $(DBG_EXECUTABLE)
