CC=gcc
CXX=g++
CFLAGS  = -O2
TARGET  = $(basename $(NAME))
SRCS    = $(NAME)
OBJDIR = ./obj/
OBJS    = $(patsubst %.C,$(OBJDIR)%.o,$(SRCS))
INCDIR  = -I./include
ROOTFLAGS = $(shell root-config --cflags)
ROOTLIBS = $(shell root-config --libs)
ROOTGLIBS = $(shell root-config --glibs)
CXXFLAGS = -Wall $(ROOTFLAGS) 
CXXLIBS = $(ROOTLIBS)

.PHONY: all
all: $(OBJDIR) $(TARGET)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)%.o: %.C
	$(CXX) $(CFLAGS) $(INCLUDE) -c -o $@ $< $(CXXLIBS) $(CXXFLAGS) -lSpectrum -lMathMore

$(TARGET): $(OBJS)
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ $^ $(CXXLIBS) $(CXXFLAGS) -lSpectrum -lMathMore

.PHONY: clean
clean:
	rm -f *.d $(OBJDIR)*.o $(TARGET)