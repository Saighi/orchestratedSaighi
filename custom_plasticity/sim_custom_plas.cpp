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
using namespace auryn;

namespace po = boost::program_options;
namespace mpi = boost::mpi;



int main(int ac, char* av[]) 
{
	bool save = false;
	
	AurynWeight wext = 0.1;
	AurynWeight wixt = 0.1;
	AurynWeight wie = 0.1;
	AurynWeight wei = 0.1;
	AurynWeight wee = 0.1;

	float rate_one = 4;
	float rate_two = 4;
	float simtime = 100;
	int errcode = 0;
	int nb_stim = 1 ;
	int nb_exc =1 ;

	AurynFloat sparseness_se = 0.2;
	AurynFloat sparseness_ie = 0.2;
	AurynFloat sparseness_ee = 0.2;
	
	AurynFloat eta = 1*10e-3;
	AurynFloat kappa = 0;
	AurynFloat tau_stdp = 20e-3;
	AurynFloat alpha = 4;
	AurynWeight maxweight = 5;
	std::string dir = "";
	std::string file_prefix = "rf1";
	char strbuf [255];

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("wext", po::value<double>(), "recurrent weight (ext)")
			("wixt", po::value<double>(), "recurrent weight to inhibition (ext)")
			("sparseness_se", po::value<double>(), "sparseness (s->e)")
			("wie", po::value<double>(), "recurrent weight (i->e)")
			("wei", po::value<double>(), "recurrent weight (e->i)")
			("wee", po::value<double>(), "recurrent weight (e->e)")
			("sparseness_ie", po::value<double>(), "sparseness (i->e)")
			("sparseness_ee", po::value<double>(), "sparseness (e->e)")
            ("simtime", po::value<double>(), "simulation time")
            ("rate_one", po::value<double>(), "background rate of input one")
			("nb_stim", po::value<double>(), "number of neurons from stimulation")
			("nb_exc", po::value<double>(), "number of neurons from excitatory population")
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
		if (vm.count("wixt")) {
			wixt = vm["wixt"].as<double>();
        }
		if (vm.count("wie")) {
			wie = vm["wie"].as<double>();
        }
		if (vm.count("wei")) {
			wei = vm["wei"].as<double>();
        }
		if (vm.count("wee")) {
			wee = vm["wee"].as<double>();
        }
		if (vm.count("sparseness_se")) {
			sparseness_se = vm["sparseness_se"].as<double>();
        }
		if (vm.count("sparseness_ie")) {
			sparseness_ie = vm["sparseness_ie"].as<double>();
        }
		if (vm.count("sparseness_ee")) {
			sparseness_ee = vm["sparseness_ee"].as<double>();
        } 

        if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
        } 

        if (vm.count("rate_one")) {
			rate_one = vm["rate_one"].as<double>();
        }

        if (vm.count("nb_stim")) {
			nb_stim = vm["nb_stim"].as<double>();
        } 
		if (vm.count("nb_exc")) {
			nb_exc = vm["nb_exc"].as<double>();
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

	PoissonGroup * PG_1 = new PoissonGroup(nb_stim,rate_one);
	//PoissonGroup * PG_2 = new PoissonGroup(1,rate_two);

	AIFGroup * neurons_e = new AIFGroup(nb_exc);
	neurons_e->dg_adapt1  = 0.1;
	neurons_e->set_tau_gaba(10e-3);
	neurons_e->set_tau_nmda(100e-3);
	neurons_e->set_ampa_nmda_ratio(0.2);

	IFGroup * neurons_i = new IFGroup(nb_exc/4);
	neurons_e->set_tau_gaba(10e-3);
	neurons_e->set_tau_nmda(100e-3);
	neurons_e->set_ampa_nmda_ratio(0.3);

	SymmetricSTDPConnection * SEE = new SymmetricSTDPConnection(PG_1,neurons_e,wext,sparseness_se,eta,kappa,tau_stdp,maxweight,GLUT,string("STDPConnection"));
	
	GlobalPFConnection * SIE;
	SIE = new GlobalPFConnection(neurons_i,neurons_e,
			wie,sparseness_ie,
			10.0,
			eta/50,
			alpha,
			50,
			GABA
			);
	SIE->set_name("IE");

	STPConnection * con_ei = new STPConnection(neurons_e,neurons_i,wei,sparseness_ie,GLUT);
	con_ei->set_tau_d(0.15);
	con_ei->set_tau_f(0.6);
	con_ei->set_ujump(0.2);

	STPConnection * con_ee = new STPConnection(neurons_e,neurons_e,wee,sparseness_ee,GLUT);
	con_ee->set_tau_d(0.15);
	con_ee->set_tau_f(0.6);
	con_ee->set_ujump(0.2);

	STPConnection * con_si = new STPConnection(PG_1,neurons_i,wixt,sparseness_se,GLUT);
	con_si->set_tau_d(0.15);
	con_si->set_tau_f(0.6);
	con_si->set_ujump(0.2);


	STPConnection * con_ii = new STPConnection(neurons_i,neurons_i,wie,sparseness_ie,GLUT);
	con_ii->set_tau_d(0.15);
	con_ii->set_tau_f(0.6);
	con_ii->set_ujump(0.2);




	sprintf(strbuf, "%s/%s.%d.sse", dir.c_str(), file_prefix.c_str(),sys->mpi_rank());
	WeightMonitor *wmon = new WeightMonitor(SEE,0,1, string(strbuf), 0.001,DATARANGE);


	sprintf(strbuf, "%s/%s.%d.sie", dir.c_str(), file_prefix.c_str(),sys->mpi_rank());
	WeightMonitor *wmon_ie = new WeightMonitor(SIE,0,1, string(strbuf), 0.001,DATARANGE);

	sprintf(strbuf, "%s/%s.%d.e.prate", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	PopulationRateMonitor * pmon_e = new PopulationRateMonitor( neurons_e, string(strbuf), 0.001 );

	sprintf(strbuf, "%s/%s.%d.i.prate", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	PopulationRateMonitor * pmon_i = new PopulationRateMonitor( neurons_i, string(strbuf), 0.001 );


	//RateChecker * chk = new RateChecker( neurons_e , -1 , 20. , 0.1);


	logger->msg("Main simtime ...",PROGRESS,true);
	if (!sys->run(simtime,false)) 
			errcode = 1;

	if ( save ) {
		sys->set_output_dir(dir);
		sys->save_network_state(file_prefix);
	}

	if (errcode) auryn_abort(errcode);

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();

	return errcode;

}

