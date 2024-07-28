#include <cmath>
#include <iostream>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include "linalg.h"
#include "NN.h"


int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<vec<double>> read_mnist(const char* path){
    std::ifstream file(path, std::ios_base::binary);
    std::vector<vec<double>> data;
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        

        vec<double> curr(n_cols * n_rows);
        for (int i = 0; i < number_of_images; ++i)
        {
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char tmp;
                    file.read((char*)&tmp, sizeof(tmp));
                    curr.data()[r * n_cols + c] = (double)tmp;
                }
            }
            data.push_back(curr);
        }
        file.close();
    }
    return data;
}

std::vector<vec<double>> read_mnist_labels(const char* path) {
    std::ifstream file(path, std::ios_base::binary);
    std::vector<vec<double>> data;
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
      
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);


        for (int i = 0; i < number_of_images; ++i)
        {
            vec<double> curr = vec<double>::filled(10);
            char tmp;
            file.read((char*)&tmp, sizeof(tmp));
            curr.data()[tmp] = 1.0;
            data.emplace_back(curr);
        }
    }
    return data;
}



void RenderData(GLuint texture) {
    if (ImGui::Begin("Image")) {
        ImGui::Image((ImTextureID)texture, ImVec2(500,500));
        ImGui::End();
    }
}


int main(){
    auto dt_test = read_mnist("t10k-images.idx3-ubyte");
    auto dt_test_label = read_mnist_labels("t10k-labels.idx1-ubyte");
    auto dt_train = read_mnist("train-images.idx3-ubyte");
    auto dt_train_label = read_mnist_labels("train-labels.idx1-ubyte");
    for (int i = 0; i < dt_train.size(); ++i) {
        dt_train[i] = dt_train[i] / 255.f;
    }
    for (int i = 0; i < dt_test.size(); ++i) {
        dt_test[i] = dt_test[i] / 255.f;
    }
    //vec<double> mX = mean(dt_test);
    //vec<double> stdX = stdev(dt_test);
    //for (int i = 0; i < dt_test.size(); ++i) {
    //    dt_test[i] = (dt_test[i] - mX) / stdX;
    //}

    Data<double> data;
    data.load(dt_train, dt_train_label);

    Network network(new LossSoftmaxCrossentropy());
    network.addLayer(new Dense(28*28, 100, new SGD()));
    network.addLayer(new ReLU());
    network.addLayer(new Dense(100, 10, new SGD()));



    network.train(data, 10000);
    std::vector<vec<double>> Y_nn;
    
    int accuracy = 0;
    for (int i = 0; i < dt_test.size(); i++) {
        accuracy += network.predict(dt_test[i]).index_maximum() == dt_test_label[i].index_maximum();
    }
    printf("accuracy: %d\n", accuracy);
    

    if (!glfwInit())
        return 1;

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glewInit();
    glfwSwapInterval(1); // Enable vsync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    ImGui::StyleColorsDark();

    ImPlot::CreateContext();
        
    const char* glsl_version = "#version 130";

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 28, 28, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenerateMipmap(GL_TEXTURE_2D);

    int id = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        
        ImGui::Begin("image");
        if (ImGui::DragInt("label", &id, 0, dt_train.size())) {
            vec<unsigned char> a(28 * 28);
            for (int i = 0; i < 28 * 28; ++i) {
                a.data()[i] = (unsigned char)dt_train[id].data()[i] * 255;
            }
            dt_train_label[id].print();
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 28, 28, 0, GL_RED, GL_UNSIGNED_BYTE, a.data());
        }
        ImGui::End();

        RenderData(texture);

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}