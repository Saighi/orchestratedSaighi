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
#include "../src/GlobalPFConnection.h"
#include "Connection_BMC.h"
using namespace auryn;

namespace po = boost::program_options;
namespace mpi = boost::mpi;



int main(int ac, char* av[]) 
{
	bool save = false;
	
	AurynWeight wie = 0.05;
	AurynWeight sigma_ext;
	AurynWeight gamma_ext;
	AurynWeight wext=0.1;

	float simtime = 100;
	int errcode = 0;
	int nb_stim = 1 ;
	int nb_inh = 20;
	
	AurynFloat eta = 1*10e-3;
	AurynFloat kappa = 0;
	AurynFloat tau_stdp = 20e-3;
	AurynFloat alpha = 4;
	AurynWeight maxweight = 0.25;
	std::string dir = "";
	std::string file_prefix = "rf1";
	char strbuf [255];

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("wext", po::value<double>(), "recurrent weight (ext)")
			("wie", po::value<double>(), "recurrent weight (i->e)")
			("gamma_ext", po::value<double>(), "gamma of weight (e->ext)")
			("sigma_ext", po::value<double>(), "sigma of weights (e->ext) distribution")
            ("simtime", po::value<double>(), "simulation time")
			("nb_stim", po::value<double>(), "number of neurons from stimulation")
			("dir", po::value<string>(), "output dir")

        ;
        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

		if (vm.count("dir")) {
			dir = vm["dir"].as<string>();
        } 
        if (vm.count("wext")) {
			wext = vm["wext"].as<double>();
        }
		if (vm.count("wie")) {
			wie = vm["wie"].as<double>();
        }
		if (vm.count("gamma_ext")) {
			gamma_ext = vm["gamma_ext"].as<double>();
        }
		if (vm.count("sigma_ext")) {
			sigma_ext = vm["sigma_ext"].as<double>();
        }
        if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
        }
        if (vm.count("nb_stim")) {
			nb_stim = vm["nb_stim"].as<double>();
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

	sprintf(strbuf, "%s/%s", dir.c_str(),"spiketrains");
	FilesInputGroup * stimgroup = new FilesInputGroup(nb_stim,string(strbuf),1);

	PoissonGroup * inh_stimgroup = new PoissonGroup(nb_inh,10);

    AIF2Group * neurons_e = new AIF2Group(1);
	neurons_e->dg_adapt1  = 0.1;
	neurons_e->dg_adapt2  = 0;
	neurons_e->set_tau_gaba(10e-3);
	neurons_e->set_tau_nmda(100e-3);
	neurons_e->set_ampa_nmda_ratio(0.2);

	

	float weight = maxweight/2; // conductance amplitude in units of leak conductance
    Connection_BMC *conStim = new Connection_BMC(stimgroup, neurons_e, weight, 1, eta,
                                               kappa,
                                               maxweight);


    conStim->set_transmitter(AMPA);

	GlobalPFConnection * SIE;
	SIE = new GlobalPFConnection(inh_stimgroup,neurons_e,
			wie,1,
			10.0,
			eta/50,
			alpha,
			50,
			GABA
			);
	SIE->set_name("IE");


    //SpikeMonitor *input_spike_mon = new SpikeMonitor(stimgroup, sys->fn("input", "ras"));
    BinarySpikeMonitor *output_spike_mon = new BinarySpikeMonitor(neurons_e, sys->fn("output", "ras"));
    WeightMonitor *wmon = new WeightMonitor(conStim, sys->fn("output", "weight"), 0.1);
    wmon->add_equally_spaced(nb_stim);

	WeightMonitor *WSIE = new WeightMonitor(SIE, sys->fn("inh", "weight"), 0.1);
    WSIE->add_equally_spaced(nb_inh);

    VoltageMonitor * output_voltage_mon = new VoltageMonitor( neurons_e, 0, sys->fn("output","mem"),0.001 );
    PopulationRateMonitor *pmon = new PopulationRateMonitor(neurons_e, sys->fn("output", "prate"), 0.01);
    //PopulationRateMonitor *pmonExt = new PopulationRateMonitor(kappa, sys->fn("outputExt", "prate"), 0.001);
    StateMonitor *nmdaMon = new StateMonitor(neurons_e, 0, "g_nmda", sys->fn("g_nmda", "state"), 0.001);
    StateMonitor *adapt2mon = new StateMonitor(neurons_e, 0, "g_adapt2", sys->fn("g_adapt2", "state"), 0.001);
    StateMonitor *adapt1mon = new StateMonitor(neurons_e, 0, "g_adapt1", sys->fn("g_adapt1", "state"), 0.001);
    StateMonitor *ampamon = new StateMonitor(neurons_e, 0, "g_ampa", sys->fn("g_ampa", "state"), 0.001);
    StateMonitor *patmon = new StateMonitor(neurons_e, 0, "in_pattern",sys->fn("in_pattern", "state"), 0.001);




	logger->msg("Main simtime ...",PROGRESS,true);
	if (!sys->run(simtime,false)) 
			errcode = 1;

	// if ( save ) {
	// 	sys->set_output_dir(dir);
	// 	sys->save_network_state(file_prefix);
	// }

	if (errcode) auryn_abort(errcode);

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();

	return errcode;

}

