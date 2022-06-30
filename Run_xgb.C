

void Run_xgb(){
    
    gROOT->ProcessLine("gSystem->Load(\"/usr/local/onnxruntime/onnxruntime-linux-x64-1.11.1/lib/libonnxruntime.so\");");
    gROOT->ProcessLine("gInterpreter->AddIncludePath(\"-I/usr/local/onnxruntime/onnxruntime-linux-x64-1.11.1/include/\");");
    gROOT->ProcessLine("#include <onnxruntime_cxx_api.h>");
    gROOT->ProcessLine("#include \"Inference_xgb.C\"");
    gROOT->ProcessLine("Inference_xgb()");
    
    return 0;
}

