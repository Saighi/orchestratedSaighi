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
#include "P10Connection.h"
using namespace auryn;

namespace po = boost::program_options;
namespace mpi = boost::mpi;



int main(int ac, char* av[]) 
{
	bool save = false;
	
	
	AurynWeight wixt = 0.1;
	AurynWeight wie = 0.1;
	AurynWeight wei = 0.1;
	AurynWeight wii = 0.1;
	AurynWeight wee = 0.1;
	AurynWeight sigma_ext;
	AurynWeight gamma_ext;
	AurynWeight wext=0.1;

	float rate_one = 4;
	float rate_two = 4;
	float simtime = 100;
	int errcode = 0;
	int nb_stim = 1 ;
	int nb_exc =1 ;

	AurynFloat sparseness_se = 0.2;
	AurynFloat sparseness_ie = 0.2;
	AurynFloat sparseness_ee = 0.2;
	AurynFloat sparseness_si = 0.2;
	
	AurynFloat eta = 1*10e-3;
	AurynFloat kappa = 0;
	AurynFloat tau_stdp = 20e-3;
	AurynFloat alpha = 1;
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
			("wii", po::value<double>(), "recurrent weight (i->i)")
			("gamma_ext", po::value<double>(), "gamma of weight (e->ext)")
			("sigma_ext", po::value<double>(), "sigma of weights (e->ext) distribution")
			("sparseness_ie", po::value<double>(), "sparseness (i->e)")
			("sparseness_ee", po::value<double>(), "sparseness (e->e)")
			("sparseness_si", po::value<double>(), "sparseness (s->i)")
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
		if (vm.count("wii")) {
			wii = vm["wii"].as<double>();
        }
		if (vm.count("gamma_ext")) {
			gamma_ext = vm["gamma_ext"].as<double>();
        }
		if (vm.count("sigma_ext")) {
			sigma_ext = vm["sigma_ext"].as<double>();
        }
		if (vm.count("sparseness_se")) {
			sparseness_se = vm["sparseness_se"].as<double>();
        }
		if (vm.count("sparseness_si")) {
			sparseness_si = vm["sparseness_si"].as<double>();
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

	sprintf(strbuf, "%s/%s", dir.c_str(),"spiketrains");
	FilesInputGroup * PG_1 = new FilesInputGroup(nb_stim,string(strbuf),1);
	//PoissonGroup * PG_2 = new PoissonGroup(1,rate_two);

	AIF2Group * neurons_e = new AIF2Group(nb_exc);
	neurons_e->dg_adapt1  = 0.1;
	neurons_e->set_tau_gaba(10e-3);
	neurons_e->set_tau_nmda(100e-3);
	neurons_e->set_ampa_nmda_ratio(0.2);

	IFGroup * neurons_i = new IFGroup(nb_exc/4);
	neurons_e->set_tau_gaba(10e-3);
	neurons_e->set_tau_nmda(100e-3);
	neurons_e->set_ampa_nmda_ratio(0.3);

	// SymmetricSTDPConnection * SEE = new SymmetricSTDPConnection(PG_1,neurons_e,wext,sparseness_se,eta,kappa,tau_stdp,maxweight,GLUT,string("STDPConnection"));
	// SEE->random_data_lognormal(gamma_ext, sigma_ext);

	P10Connection * SEE;
	SEE = new P10Connection(PG_1,neurons_e,
			wext,sparseness_se,
			eta,
			kappa,
			maxweight,
			0 // delay
			);

	SEE->set_transmitter(AMPA);
	SEE->set_name("SE");
	// STP parameters
	SEE->set_tau_d(0.15);
	SEE->set_tau_f(0.6);
	SEE->set_ujump(0.2);
	SEE->set_urest(0.2);
	SEE->random_data_lognormal(gamma_ext, sigma_ext);

	// STPConnection * SEE = new STPConnection(PG_1,neurons_e,wext,sparseness_se,GLUT);
	// SEE->set_tau_d(0.15);
	// SEE->set_tau_f(0.6);
	// SEE->set_ujump(0.2);
	// SEE->random_data_lognormal(gamma_ext, sigma_ext);

	GlobalPFConnection * SIE;
	SIE = new GlobalPFConnection(neurons_i,neurons_e,
			wie,sparseness_ie,
			10.0,
			eta/50, 
			alpha,
			5,
			GABA
			);
	SIE->set_name("IE");
		
	// STPConnection * SIE = new STPConnection(neurons_i,neurons_e,wie,sparseness_ie);
	// SIE->set_tau_d(0.15);
	// SIE->set_tau_f(0.6);
	// SIE->set_ujump(0.2);

	STPConnection * con_ei = new STPConnection(neurons_e,neurons_i,wei,sparseness_ie,GLUT);
	con_ei->set_tau_d(0.15);
	con_ei->set_tau_f(0.6);
	con_ei->set_ujump(0.2);

	STPConnection * con_ee = new STPConnection(neurons_e,neurons_e,wee,sparseness_ee,GLUT);
	con_ee->set_tau_d(0.15);
	con_ee->set_tau_f(0.6);
	con_ee->set_ujump(0.2);

	STPConnection * con_si = new STPConnection(PG_1,neurons_i,wixt,sparseness_si,GLUT);
	con_si->set_tau_d(0.15);
	con_si->set_tau_f(0.6);
	con_si->set_ujump(0.2);
	con_si->random_data_lognormal(gamma_ext, sigma_ext);

	double geta = -eta*1e-4;
	SparseConnection * con_ii = new SparseConnection(neurons_i,neurons_i,wii,sparseness_ie,GABA);

	sprintf(strbuf, "%s/%s.%d.sse", dir.c_str(), file_prefix.c_str(),sys->mpi_rank());
	WeightMonitor *wmon = new WeightMonitor(SEE, string(strbuf), 0.01);
	wmon->add_equally_spaced(1000);

	sprintf(strbuf, "%s/%s.%d.sie", dir.c_str(), file_prefix.c_str(),sys->mpi_rank());
	WeightMonitor *wmon_ie = new WeightMonitor(SIE, string(strbuf), 0.01);
	wmon_ie->add_equally_spaced(1000);

	sprintf(strbuf, "%s/%s.%d.e.prate", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	PopulationRateMonitor * pmon_e = new PopulationRateMonitor( neurons_e, string(strbuf), 0.001 );

	sprintf(strbuf, "%s/%s.%d.i.prate", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	PopulationRateMonitor * pmon_i = new PopulationRateMonitor( neurons_i, string(strbuf), 0.001 );

	sprintf(strbuf, "%s/%s.%d.e.g_adapt2", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    StateMonitor *adaptMon = new StateMonitor(neurons_e, 0, "g_adapt2", string(strbuf), 0.0001);

	sprintf(strbuf, "%s/%s.%d.e.spk", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	BinarySpikeMonitor * smon_e = new BinarySpikeMonitor( neurons_e, string(strbuf), nb_exc );


	sprintf(strbuf, "%s/%s.%d.i2.spk", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	BinarySpikeMonitor * smon_i2 = new BinarySpikeMonitor( neurons_i, string(strbuf), nb_exc/4 );

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

