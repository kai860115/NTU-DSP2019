.PHONY: all clean

CXX      = g++
CXXFLAGS = -std=c++11 -O3

TARGET   = train test cal_acc

all: $(TARGET) 

clean:
	rm -f $(TARGET)   

train: train.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

test: test.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

cal_acc: cal_acc.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

run: runtrain runtest runacc

runtrain: train 
	./train 100 model_init.txt seq_model_01.txt model_01.txt
	./train 100 model_init.txt seq_model_02.txt model_02.txt
	./train 100 model_init.txt seq_model_03.txt model_03.txt
	./train 100 model_init.txt seq_model_04.txt model_04.txt
	./train 100 model_init.txt seq_model_05.txt model_05.txt

runtest: test
	./test modellist.txt testing_data1.txt result1.txt
	./test modellist.txt testing_data2.txt result2.txt

runacc: cal_acc
	./cal_acc result1.txt testing_answer.txt acc.txt

