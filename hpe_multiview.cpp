#include "hpe_problem.h"

#include <iostream>
#include <memory>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace fs = boost::filesystem;
namespace po = boost::program_options;


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
    double dShapeWeight, dExprWeight;

    opts.add_options()
        ("project_path", po::value<string>(&projectPath)->default_value(
            R"(./data/project)"), 
            "Folder containing images and camera information.")
        ("bfm_h5_path", po::value<string>(&bfmH5Path)->default_value(
            R"(./data/BaselFaceModel_mod.h5)"), 
            "Path of Basel Face Model.")
        ("landmark_idx_path", po::value<string>(&landmarkIdxPath)->default_value(
            R"(./data/example_landmark_68.anl)"), 
            "Path of corresponding between dlib and model vertex index.")
        ("dlib_landmark_det_path", po::value<string>(&dlibLandmarkDetPath)->default_value(
            R"(./data/shape_predictor_68_face_landmarks.dat)"), 
            "Path of shape_predictor_68_face_landmarks.dat.")
        ("shape", po::value<double>(&dShapeWeight)->default_value(10.0), "Weight of shape regular term")        
        ("expr", po::value<double>(&dExprWeight)->default_value(0.001), "Weight of expression regular term")   
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
    LOG(INFO) << "\tProject path:\t\t\t" << projectPath;
    LOG(INFO) << "\tBfm path:\t\t\t" << bfmH5Path;
    LOG(INFO) << "\tLandmark indices path:\t\t" << landmarkIdxPath;
    LOG(INFO) << "\tDlib landmark detector path:\t" << dlibLandmarkDetPath;
    LOG(INFO) << "\n";

	std::unique_ptr<MHPEProblem> pHpeProblem(new MHPEProblem(projectPath, bfmH5Path, landmarkIdxPath, dlibLandmarkDetPath));
	std::shared_ptr<BfmManager>& pBfmManager = pHpeProblem->getBfmManager();

    pHpeProblem->solve(SolveExtParamsMode_Default, dShapeWeight, dExprWeight);

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
