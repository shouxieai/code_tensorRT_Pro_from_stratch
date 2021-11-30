#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>

using namespace std;

// the interface for app
int app_yolo();
int app_alphapose();
int app_fall_recognize();

void test_all(){
    app_yolo();
    app_alphapose();
    app_fall_recognize();
    INFO("test done");
}

int main(int argc, char** argv){
    const char* method = "yolo";//todo const char* 
    if (argc > 1){
        method = argv[1];
    }

    if (strcmp(method, "yolo") == 0){
        app_yolo();
    }else if(strcmp(method, "alphapose") == 0){ 
        app_alphapose();
    }else if(strcmp(method, "fall_recognize") == 0){//todo strcmp
        app_fall_recognize();
    }else{
        printf("Unknown method: %s\n", method);
        printf(
            "Help: \n"
            "    ./pro method[yolo、alphapose、fall、retinaface、arcface、arcface_video、arcface_tracker]\n"
            "\n"
            "    ./pro yolo\n"
            "    ./pro alphapose\n"
            "    ./pro fall\n"
        );
    }

    return 0;

}