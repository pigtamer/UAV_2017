del /Q .\nega\* .\pos\* .\*.yml
python exSample.py
REM train_HOG -d -dw=64 -dh=64 -pd=./pos -nd=./nega -tv=./Video_37.avi -fn=detector