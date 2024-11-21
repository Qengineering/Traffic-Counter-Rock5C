// 10-01-2024 Q-engineering

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <gflags/gflags.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include "BYTETracker.h"
#include "Tjson.h"
#include "MQTT.h"
#include "MJPGthread.h"
#include "TChannel.h"
#include <chrono>
#include <thread>
#include "postprocess.h"
#include "rknn_api.h"

using namespace cv;
using namespace std;

//----------------------------------------------------------------------------------------
// Define the command line flags
//----------------------------------------------------------------------------------------

DEFINE_bool(debug, false, "shows debug output");

//----------------------------------------------------------------------------------
Tjson         Js;
BYTETracker   Btrack;
size_t        FrameCnt=0;
size_t        FPS;
//----------------------------------------------------------------------------------
const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};
//----------------------------------------------------------------------------------
// Message handler for the MQTT subscription for any desired control channel topic
int handleMQTTControlMessages(void *context, char *topicName, int topicLen, MQTTClient_message *message)
{
    printf("MQTT message arrived\n");
    printf("     topic: %s\n", topicName);
    printf("   message: %.*s\n\n", message->payloadlen, (char*)message->payload);

    return 1;
}
//----------------------------------------------------------------------------------
// Function to read binary file into a buffer allocated in memory
static unsigned char* load_model(const char* filename, int& fileSize) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate); // Open file in binary mode and seek to the end

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }

    fileSize = (int) file.tellg(); // Get the file size
    file.seekg(0, std::ios::beg); // Seek back to the beginning

    char* buffer = (char*)malloc(fileSize); // Allocate memory for the buffer

    if (!buffer) {
        std::cerr << "Memory allocation failed." << std::endl;
        return nullptr;
    }

    file.read(buffer, fileSize); // Read the entire file into the buffer
    file.close(); // Close the file

    return (unsigned char*) buffer;
}
//----------------------------------------------------------------------------------
void DrawFrame(cv::Mat &frame,const TChannel &channel,const vector<bbox_t> &OutputTracks)
{
    if(FLAGS_debug){
        for(size_t i=0; i<channel.MapList.size(); i++){
            if(channel.MapList[i].Tframe==FrameCnt){
                Scalar s = Btrack.get_color(channel.MapList[i].Track);
                if(channel.MapList[i].Tdone){
                    rectangle(frame, Rect(channel.MapList[i].Box.l, channel.MapList[i].Box.t, channel.MapList[i].Box.r-channel.MapList[i].Box.l, channel.MapList[i].Box.b-channel.MapList[i].Box.t), s, -1);
                }
                else {
                    rectangle(frame, Rect(channel.MapList[i].Box.l, channel.MapList[i].Box.t, channel.MapList[i].Box.r-channel.MapList[i].Box.l, channel.MapList[i].Box.b-channel.MapList[i].Box.t), s, 2);
                    line(frame, Point(channel.MapList[i].Box.a, channel.MapList[i].Box.b), Point(channel.MapList[i].Box_T0.a, channel.MapList[i].Box_T0.b), cv::Scalar(0, 0, 255), 2);
                }
            }
        }
        line(frame, Point(Js.X1, Js.Y1), Point(Js.X2, Js.Y2), cv::Scalar(0, 0, 255), 2);
    }
    else{
        if(Js.Annotate){
            for(size_t i=0; i<OutputTracks.size(); i++){
                Scalar s = Btrack.get_color(OutputTracks[i].track_id);
    //                    putText(frame, format("%d", OutputTracks[i].track_id), Point(OutputTracks[i].x, OutputTracks[i].y - 5),
    //                                0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                rectangle(frame, Rect(OutputTracks[i].x, OutputTracks[i].y, OutputTracks[i].w, OutputTracks[i].h), s, 2);
            }
            line(frame, Point(Js.X1, Js.Y1), Point(Js.X2, Js.Y2), cv::Scalar(0, 0, 255), 2);
        }
    }

    int p=0;
    for(int i=0; i<10; i++){
        if(channel.CntAr[i]>0){
            putText(frame, cv::format("%s = %i", class_names[i], channel.CntAr[i]),cv::Point(10,p*20+20),cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255));
            p++;
        }
    }
}
//----------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    rknn_context   ctx;
    int            img_width          = 0;
    int            img_height         = 0;
    const float    nms_threshold      = NMS_THRESH;
    const float    box_conf_threshold = BOX_THRESH;
    int            ret;

    size_t         FrameCnt_T1;
    size_t         FPS_T1 = 15;
    int            Mcnt = 0;
    MJPGthread    *MJ = nullptr;
    TChannel       chn;
    VideoCapture   cap;

    // Parse the flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Load JSON
    // Js takes care for printing errors.
    Js.LoadFromFile("./config.json");
    bool Success = Js.GetSettings();
    if(!Success){
        cout << "Error loading settings from the json file!\n\nPress <Ctrl>+<C> to quit" << endl;
        while(1) sleep(10);
        return 0;
    }
    if(Js.Version != VERSION){
        cout << "Wrong version of the json file!" << endl;
        cout << "\nSoftware version is " << VERSION << endl;
        cout << "JSON file is " << Js.Version << endl;
        cout << "\nPress <Ctrl>+<C> to quit" << endl;
        while(1) sleep(10);
        return 0;
    }
    cout << "Loaded JSON" << endl;

    //load p1 and p2 in the mapper
    chn.Init();

    //export MQTT environment
    setenv("MQTT_SERVER", Js.MQTT_Server.c_str(), 1);       // The third argument '1' specifies to overwrite if it already exists
    setenv("MQTT_CLIENT_ID", Js.MQTT_Client.c_str(), 1);

    // Connect MQTT messaging
    if(mqtt_start(handleMQTTControlMessages) != 0) {
        cout << "MQTT NOT started: have you set the ENV varables?\nPress <Ctrl>+<C> to quit" << endl;
        while(1) sleep(10);
        return 0;
    }
    mqtt_connect();
	mqtt_subscribe("traffic");

	//load MJPEG_Port
    MJ = new MJPGthread();
    if(Js.MJPEG_Port>7999){
        MJ->Init(Js.MJPEG_Port);
        cout << "Opened MJpeg port: " << Js.MJPEG_Port << endl;
    }

    // Create the neural network
    printf("Loading mode...\n");
    int            model_data_size = 0;
    unsigned char* model_data      = load_model(Js.npuMstr.c_str(), model_data_size);

    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    //no need to hold the model in the buffer. Get some memory back
    free(model_data);

    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for(uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    int channel = 3;
    int width   = 0;
    int height  = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        channel = input_attrs[0].dims[1];
        height  = input_attrs[0].dims[2];
        width   = input_attrs[0].dims[3];
    }
    else {
        height  = input_attrs[0].dims[1];
        width   = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_UINT8;
    inputs[0].size         = width * height * channel;
    inputs[0].fmt          = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    // You may not need resize when src resulotion equals to dst resulotion
    cv::Mat orig_img;
    cv::Mat img;
    cv::Mat resized_img;

    cap.open(Js.Stream);
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the stream" << endl;
        return 0;
    }
    int fps = cap.get(CAP_PROP_FPS);

    Btrack.Init(fps, 30);

    printf("Start grabbing, press ESC on Live window to terminated...\n");

    // Capture the start time
    auto lastRunTime = std::chrono::steady_clock::now();
    FrameCnt_T1=0;

    // You may not need resize when src resulotion equals to dst resulotion
    while(1){
        cap >> orig_img;
        if(orig_img.empty()) {
            printf("Error grabbing\n");
            break;
        }

        FrameCnt++;

        cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
        img_width  = img.cols;
        img_height = img.rows;

        //check sizes
        if (img_width != width || img_height != height) {
            cv::resize(img,resized_img,cv::Size(width,height));
            inputs[0].buf = (void*)resized_img.data;
        } else {
            inputs[0].buf = (void*)img.data;
        }

        rknn_inputs_set(ctx, io_num.n_input, inputs);

        rknn_output outputs[io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (uint32_t i = 0; i < io_num.n_output; i++) {
            outputs[i].index = i;
            outputs[i].want_float = 0;
        }

        rknn_run(ctx, NULL);
        rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

        // post process
        float scale_w = (float)width / img_width;
        float scale_h = (float)height / img_height;

        detect_result_group_t detect_result_group;
        std::vector<float> out_scales;
        std::vector<int32_t> out_zps;
        for(uint32_t i = 0; i < io_num.n_output; ++i){
            out_scales.push_back(output_attrs[i].scale);
            out_zps.push_back(output_attrs[i].zp);
        }

        post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
                   box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);


        //convert to vector<TargetBox>
        vector<TargetBox> objects;
        for(int i=0; i<detect_result_group.count; i++){
            TargetBox o;

            o.x1     = detect_result_group.results[i].box.left;
            o.y1     = detect_result_group.results[i].box.top;
            o.x2     = detect_result_group.results[i].box.right;
            o.y2     = detect_result_group.results[i].box.bottom;
            o.obj_id = detect_result_group.results[i].obj_id;
            o.score  = detect_result_group.results[i].prop;

            objects.push_back(o);
        }

        ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

        vector<bbox_t> OutputTracks = Btrack.update(objects);

        chn.Execute(OutputTracks);

        DrawFrame(orig_img, chn, OutputTracks);

        // Check if 1 second has passed
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastRunTime).count() >= 1) {
            // get FPS
            FPS = (FrameCnt_T1<FrameCnt)? (FrameCnt-FrameCnt_T1) : FPS_T1;

            if(FLAGS_debug){
                cout << "FPS: "<< FPS << endl;
            }

            // Update last run time
            lastRunTime = now;
            FrameCnt_T1 = FrameCnt;
            FPS_T1      = FPS;

            // Check if we must send a message
            if(++Mcnt >= Js.MesSec){
                chn.SendMessage();
                Mcnt = 0;
            }

            // Check if it's midnight
            auto time_t_now = std::chrono::system_clock::to_time_t( std::chrono::system_clock::now() );
            auto localTime = *std::localtime(&time_t_now);

            if (localTime.tm_hour == 0 && localTime.tm_min == 0) {
                // Clear the array at midnight
                chn.ResetCntAr();
                // Wait a minute to avoid multiple clearings within the same minute
                std::this_thread::sleep_for(std::chrono::minutes(1));
            }
        }

        //send the overview into the world
        if(Js.MJPEG_Port>7999){
            //send the result to the socket
            cv::Mat FrameMJPEG(Js.MJPEG_Height, Js.MJPEG_Width, CV_8UC3);
            cv::resize(orig_img,FrameMJPEG,FrameMJPEG.size(),0,0);
            MJ->Send(FrameMJPEG);
        }

        //show output
//        std::cout << "FPS" << f/16 << std::endl;
        imshow("Rock 5C - 2,5 GHz - 4 Mb RAM", orig_img);
        char esc = cv::waitKey(5);
        if(esc == 27) break;

//      imwrite("./out.jpg", orig_img);
    }

    std::cout << "Closing the camera" << std::endl;
    cv::destroyAllWindows();

    // release
    ret = rknn_destroy(ctx);

    mqtt_disconnect();
    mqtt_close();

    // Clean up and exit
    gflags::ShutDownCommandLineFlags();

    std::cout << "Bye!" << std::endl;

    return 0;
}
