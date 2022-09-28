COMMON=/O2 /MT /EHsc /arch:AVX /I../include ../bin/mujoco150.lib /Fe../bin/

all:
	cl $(COMMON) basic.cpp       ../bin/glfw3.lib
	cl $(COMMON) simulate.cpp    ../bin/glfw3.lib
	cl $(COMMON) record.cpp      ../bin/glfw3.lib
	cl $(COMMON) test.cpp       
	cl $(COMMON) compile.cpp    
	cl $(COMMON) derivative.cpp  /openmp
	del *.obj
