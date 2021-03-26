#include "hpe_problem.h"

#include <iostream>
#include <chrono>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace fs = boost::filesystem;
namespace po = boost::program_options;
using namespace std;


int main(int argc, char* argv[])
{  
    // logging
    google::InitGoogleLogging(argv[0]); 
    FLAGS_logtostderr = false;
    if(fs::exists(LOG_PATH)) 
        fs::remove_all(LOG_PATH);
    fs::create_directory(LOG_PATH);
    FLAGS_alsologtostderr = true;
    FLAGS_log_dir = LOG_PATH;
    FLAGS_log_prefix = true; 
    FLAGS_colorlogtostderr =true;

    // command options
    po::options_description opts("Options");
    po::variables_map vm;

    string projectPath, bfmH5Path;
    string landmarkIdxPath, dlibLandmarkDetPath;

    opts.add_options()
        ("project_path", po::value<string>(&projectPath)->default_value(
            R"(/media/keith/SAKURA/face_zzm/project)"), 
            "Folder containing images and camera information.")
        ("bfm_h5_path", po::value<string>(&bfmH5Path)->default_value(
            R"(/home/keith/Data/BaselFaceModel_mod.h5)"), 
            "Path of Basel Face Model.")
        ("landmark_idx_path", po::value<string>(&landmarkIdxPath)->default_value(
            R"(/home/keith/Project/head-pose-estimation/data/example_landmark_68.anl)"), 
            "Path of corresponding between dlib and model vertex index.")
        ("dlib_landmark_det_path", po::value<string>(&dlibLandmarkDetPath)->default_value(
            R"(/home/keith/Data/shape_predictor_68_face_landmarks.dat)"), 
            "Path of shape_predictor_68_face_landmarks.dat.")
        ("help,h", "Help message");
    
    try
    {
        po::store(po::parse_command_line(argc, argv, opts), vm);
    }
    catch(...)
    {
        LOG(ERROR) << "These exists undefined command options.";
        return -1;
    }

    po::notify(vm);
    if(vm.count("help"))
    {
        LOG(INFO) << opts;
        return 0;
    }

    LOG(INFO) << "Check inputs:";
    LOG(INFO) << "Project path:\t" << projectPath;
    LOG(INFO) << "Bfm path:\t" << bfmH5Path;
    LOG(INFO) << "Landmark indices path:\t" << landmarkIdxPath;
    LOG(INFO) << "Dlib landmark detector path:\t" << dlibLandmarkDetPath;
    LOG(INFO) << "\n";

	MHPEProblem *pHpeProblem = new MHPEProblem(projectPath, bfmH5Path, landmarkIdxPath);
	BaselFaceModelManager *pBfmManager = pHpeProblem->getModel();

    auto start = std::chrono::system_clock::now();    // Start of solving 
    pHpeProblem->solve(SolveExtParamsMode_Default, dlibLandmarkDetPath);
    auto end = std::chrono::system_clock::now();    // End of solving
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG(INFO) << "Cost of solution: "
        << (double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den)
        << " seconds." << std::endl;            

    // Show results 
    pBfmManager->printExtParams();
    // pBfmManager->printShapeCoef();
    // pBfmManager->printExprCoef();

    // Generate whole face, because before functions only process landmarks
    // pBfmManager->genFace();

    // Write face into .ply model file
    // pBfmManager->writePly("rnd_face.ply", ModelWriteMode_CameraCoord);

    google::ShutdownGoogleLogging();
    return 0;
}
