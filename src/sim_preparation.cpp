#include <string>
#include "generate_events.cpp"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int ac, char *av[])
{

    double begin = 0; 
    double simtime = 3600; 
    int nb_neurons = 4096;
    int bgrate = 10;
    int pattern_rate = 5;
    double time_pattern=0.1;
    char strbuf [255];
    bool new_pattern = true;
    int nb_pattern = 1;

    string dir = "/users/nsr/saighi/data/sim_network/sim/";
    string file_name = "";

        try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("simtime", po::value<double>(), "simulation time")
            ("bgrate", po::value<double>(), "background rate of input")
            ("begin", po::value<double>(), "begining of pattern presentation")
            ("pattern_rate", po::value<double>(), "rate of pattern")
            ("time_pattern", po::value<double>(), "duration of pattern")
            ("nb_neurons", po::value<double>(), "number of neurones")
            ("dir", po::value<string>(), "output dir")
            ("nonew_pattern", po::value<string>(), "don't use a new set of patterns")
            ("nb_pattern", po::value<int>(), "number of different patterns presented")

        ;
        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);  

        if (vm.count("nb_pattern")) {
			nb_pattern = vm["nb_pattern"].as<int>();
        } 
        if (vm.count("nonew_pattern")) {
			new_pattern = false;
        } 
        if (vm.count("begin")) {
			begin = vm["begin"].as<double>();
        } 
        if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
        } 
        if (vm.count("bgrate")) {
			bgrate = vm["bgrate"].as<double>();
        } 
        if (vm.count("pattern_rate")) {
			pattern_rate = vm["pattern_rate"].as<double>();
        } 
        if (vm.count("time_pattern")) {
			time_pattern = vm["time_pattern"].as<double>();
        } 

        if (vm.count("nb_neurons")) {
			nb_neurons = vm["nb_neurons"].as<double>();
        }
        if (vm.count("dir")) {
			dir = vm["dir"].as<string>();
        } 

    }
    
    catch(std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
		std::cerr << "Exception of unknown type!\n";
    }



    file_name = "pattern_time";
    sprintf(strbuf, "%s/%s", dir.c_str(), file_name.c_str());
    write_random_events(string(strbuf),time_pattern, pattern_rate,
                         begin, simtime, nb_pattern);
    if (new_pattern){
        for (int i = 0; i < nb_pattern; i++) {
            file_name = "pattern";
            sprintf(strbuf, "%s/%s_%i", dir.c_str(), file_name.c_str(),i);
            generate_pattern(string(strbuf),
                            bgrate, nb_neurons, time_pattern); 
        }
    }

    generate_spiketrains("test", bgrate,
                      nb_neurons,  begin, simtime, 0.1);

}
