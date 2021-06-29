/* 
* Copyright 2015 Friedemann Zenke
*
* This file is part of Auryn, a simulation package for plastic
* spiking neural networks.
* 
* Auryn is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* Auryn is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "auryn.h"

using namespace auryn;

namespace po = boost::program_options;
namespace mpi = boost::mpi;

init_w = 0.25

int main(int ac, char* av[]) 
{

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("init_w", po::value<double>(), "recurrent weight (ext)")
            ("simtime", po::value<double>(), "simulation time")
            ("rate_one", po::value<double>(), "background rate of input one")
			("rate_two", po::value<double>(), "background rate of input two")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("wext")) {
			init_w = vm["init_w"].as<double>();;
        }

        if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
        } 

        if (vm.count("rate_one")) {
			rate_one = vm["rate_one"].as<double>();;
        }

        if (vm.count("rate_two")) {
			chain = vm["rate_two"].as<double>();;
        } 
    }
    catch(std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
		std::cerr << "Exception of unknown type!\n";
    }


	auryn_init( ac, av, dir, "sim_custom_plas", file_prefix );
	sys->set_master_seed(42);
	logger->set_logfile_loglevel(VERBOSE);
	


	sprintf(strbuf, "%s/%s.%d.sse", dir.c_str(), file_prefix.c_str(),sys->mpi_rank());
	WeightMonitor *wmon = new WeightMonitor(con_stim_e,0,taille, string(strbuf), 0.0001,DATARANGE);

	//RateChecker * chk = new RateChecker( neurons_e , -1 , 20. , 0.1);



	logger->msg("Main simtime ...",PROGRESS,true);
	if (!sys->run(simtime,false)) 
			errcode = 1;

	if ( save ) {
		sys->set_output_dir(dir);
		sys->save_network_state(file_prefix);
	}

	sprintf(strbuf, "%s/%s.%d.ext.wmat", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	con_stim_e->write_to_file(strbuf);

	if (errcode) auryn_abort(errcode);

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();

	return errcode;

}

