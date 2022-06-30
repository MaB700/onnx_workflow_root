#if !defined(__CLING__)
#include "TROOT.h"
#endif

void Run_cnn(){
    
    gROOT->ProcessLine("gSystem->Load(\"/usr/local/onnxruntime/onnxruntime-linux-x64-1.11.1/lib/libonnxruntime.so\");");
    gROOT->ProcessLine("gInterpreter->AddIncludePath(\"-I/usr/local/onnxruntime/onnxruntime-linux-x64-1.11.1/include/\");");
    gROOT->ProcessLine("#include <onnxruntime_cxx_api.h>");
    gROOT->ProcessLine("#include \"Inference_cnn.C\"");
    gROOT->ProcessLine("Inference_cnn()");
    
}

