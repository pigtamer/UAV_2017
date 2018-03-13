x=0
g++ -std=c++11 main.cpp  -o main `pkg-config --cflags --libs opencv` |2> error.log
x=`wc -l < error.log` #compile source and dump errors to a log file

if [ x=0 ] 
then
	./main
	#run program when successfully compiled
else
	echo 'Compilation failed. Check log for more info.'
fi
