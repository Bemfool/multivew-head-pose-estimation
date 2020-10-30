#include "hpe_problem.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

// #define STB_IMAGE_IMPLEMENTATION
// #include <stb_image.h>
#include <shader.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 640;
const unsigned int SCR_HEIGHT = 480;


int main(int argc, char** argv)
{  
	google::InitGoogleLogging(argv[0]);

    BFM_DEBUG(PRINT_GREEN "#################### OpenGL Init ####################\n" COLOR_END);
    // Initialize and configure GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); 
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Head Pose Estimation - Oneshot", nullptr, nullptr);
    if (window == nullptr)
    {
        BFM_ERROR("Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    // Load all OpenGL function pointers 
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        BFM_ERROR("Failed to initialize GLAD\n");
        return -1;
    }

	HeadPoseEstimationProblem *pHpeProblem = new HeadPoseEstimationProblem();
	BaselFaceModelManager *pBfmManager = pHpeProblem->getModel();
    // pBfmManager->genAvgFace();
    // pBfmManager->writePly();
    // return 0;

	try 
	{
        // Start of solving 
        auto start = std::chrono::system_clock::now();
        pHpeProblem->solve();
        
        // End of solving
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        BFM_DEBUG("Cost of solution: %lf Second\n", 
            double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);             

        // Show results 
        pBfmManager->printExtParams();
        // pBfmManager->printShapeCoef();
        // pBfmManager->printExprCoef();

        // Generate whole face, because before functions only process landmarks
        pBfmManager->genFace();

        // Write face into .ply model file
        pBfmManager->writePly("rnd_face.ply", ModelWriteMode_CameraCoord);

	} catch (exception& e) {
        BFM_ERROR("Exception thrown: %s\n", e.what());
	}

    glfwTerminate();
    return 0;
}



void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}


void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}