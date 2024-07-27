#include <cmath>
#include <iostream>

#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include "linalg.h"
#include "NN.h"


void RenderData(std::vector<vec<double>> const& X_real, std::vector<vec<double>> const& Y_real,
                std::vector<vec<double>> const& X_nn, std::vector<vec<double>> const& Y_nn) {
    std::vector < double > x_r;
    std::vector < double > y_r;
    for (int i = 0; i < X_real.size(); ++i) {
        x_r.push_back(X_real[i].data()[0]);
        y_r.push_back(Y_real[i].data()[0]);
    }

    std::vector < double > x_nn;
    std::vector < double > y_nn;
    for (int i = 0; i < X_nn.size(); ++i) {
        x_nn.push_back(X_nn[i].data()[0]);
        y_nn.push_back(Y_nn[i].data()[0]);
    }

    if (ImPlot::BeginPlot("Shaded Plots")) {
        ImPlot::PlotLine("Real", x_r.data(), y_r.data(), x_r.size());
        ImPlot::PlotLine("NN", x_nn.data(), y_nn.data(), x_nn.size());
        ImPlot::EndPlot();
    }
}


int main(){
    // Data that we will approximate
    std::vector<vec<double>> X;
    std::vector<vec<double>> Y;
    {
        vec<double> a(1);
        vec<double> b(1);
        for (int i = 0; i < 600; ++i) {
            a.data()[0] = i * 40 / 360.0 * 3.14;
            X.emplace_back(a);
            b.data()[0] = (double)(std::sin(i * 4 / 360.0 * 3.14));
            Y.emplace_back(b);
        }
    }

    // Z-Score normalize 
    vec<double> mX = mean(X);
    vec<double> stdX = stdev(X);
    for (int i = 0; i < X.size(); ++i) 
        X[i] = (X[i] - mX) / stdX;

    // Load data to Data object
    Data<double> data;
    data.load(X, Y);

    // NN initialization
    Network network(new LossMSE());
    network.addLayer(new Dense(1, 12, new SGD()));
    network.addLayer(new Sigmoid());
    network.addLayer(new Dense(12, 1, new SGD()));

    // Train
    network.train(data, 100000);

    // Predict
    std::vector<vec<double>> Y_nn;
    for (int i = 0; i < X.size(); i++)
        Y_nn.push_back(network.predict(X[i]));

    ///////////////////////////////////////////////////

    // Rendering stuff
    if (!glfwInit())
        return 1;

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Sin approximation", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; 
    ImGui::StyleColorsDark();

    ImPlot::CreateContext();
        
    const char* glsl_version = "#version 130";

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Data:");
        RenderData(X, Y, X, Y_nn);
        ImGui::End();

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