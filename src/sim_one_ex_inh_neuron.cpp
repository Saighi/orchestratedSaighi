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
#include "GlobalPFConnection.h"
#include "P10ConnectionP.h"

#define N_EXEC_WEIGHTS 800

using namespace auryn;

namespace po = boost::program_options;
namespace mpi = boost::mpi;

int main(int ac, char* av[]) 
{

	string dir = "/lcncluster/zenke/reset/";
	string file_prefix = "rc";
	string infilename = "";

	char strbuf [255];
	string msg;

	bool save = false;
	bool chain = false;
	bool prime = false;
	bool consolidation = true;
	bool isp_active = true;

	bool inh_input = false;
	bool noisy_initial_weights = false;
	bool consolidate_initial_weights = false;

	NeuronID stimsize = 1;
	NeuronID size = 1;
	NeuronID seed = 1;
	double alpha = 3;
	double kappa = 10;
	double tauf = 200e-3;
	double ujump = 0.2;
	double taud = 200e-3;
	double eta = 1e-3;

	double beta = 5.0e-2;
	double delta = 2.0e-2;
	double weight_a = 0.1;
	double weight_c = 0.5;
	double adapt = 0.0;

	double strong_weights = 0.0; // determines the fraction of weights initialized at weight_c -- FIXME not implemented

	double pot_strength = 0.1;

	double ontime = 1.0;
	double offtime = 5.0;

	double scale = 35;

	double wmax = 5.0;
	double wmin = 0.0;
	double wmaxi = 5.0;

	double bgrate = 10.0;

	int preferred = -1;

	string stimfile = ""; // stimulus patterns file
	string prefile = ""; // preload patters file
	string recfile = ""; // file with receptive fields
	string monfile = ""; // patternsto monitor file
	string patternFile = "/users/nsr/saighi/pub2015orchestrated/src/data/30hzTimePat"; // patternsto monitor file


	AurynWeight wee = 0.1;
	AurynWeight wei = 0.2;
	AurynWeight wie = 0.2;
	AurynWeight wii = 0.2;

	AurynWeight chi = 1.0;
	AurynWeight xi = 0.5;

	double sparseness = 0.1;
	double sparseness_ext = 0.05;
	double wext = 0.2;

	double simtime = 3600.;

	int errcode = 0;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("load", po::value<string>(), "input weight matrix")
            ("prefix", po::value<string>(), "set file prefix")
            ("save", "save network state at end of sim")
            ("chain", "chain mode for pattern loader")
            ("prime", "prime network with a burn-in phase")
            ("noconsolidation", "switches off consolidation")
            ("noisp", "switches off isp")
            ("noisyweights", "switches noisy initial weights on")
            ("consolidateweights", "initialize weights as consolidated")
            ("inhinput", "switches external input to inh on")
            ("alpha", po::value<double>(), "exc input rate")
            ("taud", po::value<double>(), "time constant of synaptic depression")
            ("tauf", po::value<double>(), "time constant of synaptic facilitation")
            ("ujump", po::value<double>(), "u jump STP constant")
            ("chi", po::value<double>(), "chi factor - pattern preload strength")
            ("xi", po::value<double>(), "xi factor - stimulation strength")
            ("wext", po::value<double>(), "recurrent weight (ext)")
            ("wee", po::value<double>(), "recurrent weight (wee)")
            ("wei", po::value<double>(), "recurrent weight (wei)")
            ("wii", po::value<double>(), "recurrent weight (wii)")
            ("wie", po::value<double>(), "recurrent weight (wie)")
            ("extsparse", po::value<double>(), "external sparseness")
            ("intsparse", po::value<double>(), "internal sparseness")
            ("simtime", po::value<double>(), "simulation time")
            ("ontime", po::value<double>(), "simulation time")
            ("offtime", po::value<double>(), "simulation time")
            ("dir", po::value<string>(), "output dir")
            ("eta", po::value<double>(), "the learning rate")
            ("beta", po::value<double>(), "decay parameter")
            ("potstrength", po::value<double>(), "potential strength parameter")
            ("delta", po::value<double>(), "growth parameter")
            ("weight_a", po::value<double>(), "weight_a")
            ("weight_c", po::value<double>(), "weight_c")
            ("strongw", po::value<double>(), "fraction of initially strong weights")
            ("size", po::value<int>(), "simulation size")
            ("seed", po::value<int>(), "random seed ")
            ("prefile", po::value<string>(), "preload file")
            ("recfile", po::value<string>(), "receptive field file")
            ("scale", po::value<double>(), "stimulus strength")
            ("adapt", po::value<double>(), "adaptation jump size for long time constant")
            ("bgrate", po::value<double>(), "background rate of input")
            ("preferred", po::value<int>(), "num of preferred stim")
			("patternFile", po::value<int>(), "repeated pattern")
            ("monfile", po::value<string>(), "monitor file")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
			std::cout << desc << "\n";
            return 1;
        }

        if (vm.count("load")) {
			infilename = vm["load"].as<string>();
        } 

        if (vm.count("prefix")) {
			std::cout << "simulation prefix " 
                 << vm["prefix"].as<string>() << ".\n";
			file_prefix = vm["prefix"].as<string>();
        } 

        if (vm.count("save")) {
			save = true;
        }

        if (vm.count("chain")) {
			chain = true;
        } 

        if (vm.count("prime")) {
			prime = true;
        } 

        if (vm.count("noconsolidation")) {
			consolidation = false;
        } 

        if (vm.count("noisp")) {
			isp_active = false;
        } 

        if (vm.count("noisyweights")) {
			noisy_initial_weights = true;
        } 

        if (vm.count("consolidateweights")) {
			consolidate_initial_weights = true;
        } 

        if (vm.count("inhinput")) {
			inh_input = true;
        } 

        if (vm.count("alpha")) {
			alpha = vm["alpha"].as<double>();
        } 

        if (vm.count("taud")) {
			taud = vm["taud"].as<double>();
        } 

        if (vm.count("tauf")) {
			tauf = vm["tauf"].as<double>();
        } 

        if (vm.count("ujump")) {
			ujump = vm["ujump"].as<double>();
        } 

        if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
        } 

        if (vm.count("ontime")) {
			ontime = vm["ontime"].as<double>();
        } 

        if (vm.count("offtime")) {
			offtime = vm["offtime"].as<double>();
        } 

        if (vm.count("dir")) {
			dir = vm["dir"].as<string>();
        } 

        if (vm.count("chi")) {
			chi = vm["chi"].as<double>();
        } 

        if (vm.count("xi")) {
			xi = vm["xi"].as<double>();
        } 

        if (vm.count("wext")) {
			wext = vm["wext"].as<double>();
        } 

        if (vm.count("wee")) {
			wee = vm["wee"].as<double>();
        } 

        if (vm.count("wei")) {
			wei = vm["wei"].as<double>();
        } 

        if (vm.count("wii")) {
			wii = vm["wii"].as<double>();
        } 

        if (vm.count("wie")) {
			wie = vm["wie"].as<double>();
        } 

        if (vm.count("extsparse")) {
			sparseness_ext = vm["extsparse"].as<double>();
        } 

        if (vm.count("intsparse")) {
			sparseness = vm["intsparse"].as<double>();
        } 

        if (vm.count("eta")) {
			eta = vm["eta"].as<double>();
        } 

        if (vm.count("beta")) {
			beta = vm["beta"].as<double>();
        } 

        if (vm.count("potstrength")) {
			pot_strength = vm["potstrength"].as<double>();
        } 

        if (vm.count("delta")) {
			delta = vm["delta"].as<double>();
        } 

        if (vm.count("weight_a")) {
			weight_a = vm["weight_a"].as<double>();
        } 

        if (vm.count("weight_c")) {
			weight_c = vm["weight_c"].as<double>();
        } 

        if (vm.count("strongw")) {
			strong_weights = vm["strongw"].as<double>();
        } 

        if (vm.count("size")) {
			size = vm["size"].as<int>();
        } 

        if (vm.count("stimfile")) {
			stimfile = vm["stimfile"].as<string>();
			monfile = stimfile;
        }

		if (vm.count("patternFile")) {
			patternFile = vm["patternFile"].as<string>();
        }

        if (vm.count("prefile")) {
			prefile = vm["prefile"].as<string>();
        } 

        if (vm.count("recfile")) {
			recfile = vm["recfile"].as<string>();
        } 

        if (vm.count("scale")) {
			scale = vm["scale"].as<double>();
        } 

        if (vm.count("adapt")) {
			adapt = vm["adapt"].as<double>();
        } 

        if (vm.count("bgrate")) {
			bgrate = vm["bgrate"].as<double>();
        } 

        if (vm.count("preferred")) {
			preferred = vm["preferred"].as<int>();
        } 

        if (vm.count("monfile")) {
			monfile = vm["monfile"].as<string>();
        } 

        if (vm.count("seed")) {
			seed = vm["seed"].as<int>();
        } 
    }
    catch(std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
		std::cerr << "Exception of unknown type!\n";
    }


	auryn_init( ac, av, dir, "sim_one_excitatory_neuron", file_prefix );
	sys->set_master_seed(42);
	logger->set_logfile_loglevel(VERBOSE);
	
	//log params
	logger->parameter("alpha",alpha);
	logger->parameter("beta",beta);
	logger->parameter("delta",delta);
	logger->parameter("eta",eta);
	logger->parameter("wee",wee);
	logger->parameter("wext",wext);
	logger->parameter("chi",chi);
	logger->parameter("xi",xi);

	logger->parameter("stimfile",stimfile);
	logger->parameter("monfile",monfile);
	logger->parameter("offtime",offtime);
	logger->parameter("ontime",ontime);

	logger->parameter("taud",taud);
	logger->parameter("tauf",tauf);
	logger->parameter("ujump",ujump);

	AIF2Group * neurons_e = new AIF2Group(40);
	neurons_e->dg_adapt1  = 0.1;
	neurons_e->dg_adapt2  = adapt;
	neurons_e->set_tau_gaba(10e-3);
	neurons_e->set_tau_nmda(100e-3);
	neurons_e->set_ampa_nmda_ratio(0.2);

	IFGroup * neurons_i2 = new IFGroup(200);
	neurons_i2->set_tau_ampa(5e-3);
	neurons_i2->set_tau_gaba(10e-3);
	neurons_i2->set_tau_nmda(100e-3);
	neurons_i2->set_ampa_nmda_ratio(0.3);

	STPConnection * con_ei2 = new STPConnection(neurons_e,neurons_i2,wei*3,sparseness,GLUT);
	con_ei2->set_tau_d(taud);
	con_ei2->set_tau_f(0.6);
	con_ei2->set_ujump(0.2);
	int taille = 200;
	//J'ai changé la taille
	FileInputGroup* stimgroup = new FileInputGroup(taille,"data/sim_one_ex_inh_neuron_many_input"); // A CHANGER

	double raw_delta = delta*eta/1e-3;
	double wtmax = 1.0/4*(weight_c-weight_a);
	double normalization_factor = (wtmax-weight_a)*(wtmax-(weight_a+weight_c)/2)*(wtmax-weight_c);
	double geta = -eta*1e-4;

	
	// External input
	P10ConnectionP * con_stim_e = NULL;
	sprintf(strbuf, "%s/%s.u_x", dir.c_str(), file_prefix.c_str());
	con_stim_e = new P10ConnectionP( stimgroup, neurons_e,
		wext, string(strbuf) ,sparseness_ext,
		eta,
		kappa, // supposedly deprecated
		wmax,
		GLUT
		);


	con_stim_e->set_weight_a(weight_a);
	con_stim_e->set_weight_c(weight_c);
	con_stim_e->set_tau_d(taud);
	con_stim_e->set_tau_f(tauf);
	con_stim_e->set_ujump(ujump);
	con_stim_e->set_urest(ujump);
	con_stim_e->set_beta(beta);
	con_stim_e->delta = raw_delta*eta;
	con_stim_e->set_min_weight(wmin);
	// con_stim_e->random_data(wext,wext);
	if ( noisy_initial_weights )
		con_stim_e->random_data(wext,wext);
	con_stim_e->set_name("Stim->E");
	con_stim_e->consolidation_active = consolidation;
	con_stim_e->pot_strength = pot_strength/normalization_factor;
	if ( consolidate_initial_weights )
		con_stim_e->consolidate();

	GlobalPFConnection * con_i2e;
	con_i2e = new GlobalPFConnection(neurons_i2,neurons_e,
			wie,sparseness,
			10.0,
			eta/50,
			alpha,
			wmaxi,
			GABA
			);
	con_i2e->set_name("I2E");
	con_i2e->stdp_active = isp_active;
	// SparseConnection * con_i2e = new SparseConnection(neurons_i2,neurons_e,wie,sparseness,GABA);
	// con_i2e->set_name("I2E");

	sprintf(strbuf, "%s/%s.%d.sie", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	WeightMonitor * wmonie = new WeightMonitor( con_i2e,0,40, string(strbuf), 0.0001,DATARANGE);


	sprintf(strbuf, "%s/%s.%d.sse", dir.c_str(), file_prefix.c_str(),sys->mpi_rank());
	WeightMonitor *wmon = new WeightMonitor(con_stim_e,0,taille, string(strbuf), 0.0001,DATARANGE);

  	sprintf(strbuf, "%s/%s.%d.e.mem", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	VoltageMonitor * stmon_mem = new VoltageMonitor( neurons_e, 0, string(strbuf),0.0001 );
	sprintf(strbuf, "%s/%s.%d.i.mem", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	VoltageMonitor * imon_mem = new VoltageMonitor( neurons_i2, 0, string(strbuf),0.0001 );

	//stmon_mem->record_for(10); // stops recording after 10s

	sprintf(strbuf, "%s/%s.%d.e.spk", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	BinarySpikeMonitor * smon_e = new BinarySpikeMonitor( neurons_e, string(strbuf) );

	sprintf(strbuf, "%s/%s.%d.i.spk", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	BinarySpikeMonitor * smon_i = new BinarySpikeMonitor( neurons_i2, string(strbuf) );

	sprintf(strbuf, "%s/%s.%d.ext.spk", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	BinarySpikeMonitor * smon_ext = new BinarySpikeMonitor( stimgroup, string(strbuf) );

	sprintf(strbuf, "%s/%s.%d.e.g_nmda", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    StateMonitor *nmdaMon = new StateMonitor(neurons_e, 0, "g_nmda", string(strbuf), 0.0001);

	sprintf(strbuf, "%s/%s.%d.e.g_ampa", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    StateMonitor *gampaMon = new StateMonitor(neurons_e, 0, "g_ampa", string(strbuf), 0.0001);

	sprintf(strbuf, "%s/%s.%d.e.g_gaba", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    StateMonitor *gabaMon = new StateMonitor(neurons_e, 0, "g_gaba", string(strbuf), 0.0001);

	sprintf(strbuf, "%s/%s.%d.e.g_adapt1", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    StateMonitor *adaptMon = new StateMonitor(neurons_e, 0, "g_adapt1", string(strbuf), 0.0001);

	sprintf(strbuf, "%s/%s.%d.i.g_ampa", dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    StateMonitor *gampaMonI = new StateMonitor(neurons_i2, 0, "g_ampa", string(strbuf), 0.0001);

	RateChecker * chk = new RateChecker( neurons_e , -1 , 20. , 0.1);

	// load if necessary
	if (!infilename.empty()) {
		logger->msg("Loading from file ...",PROGRESS,true);
		sys->load_network_state(infilename.c_str());

		// auryn_vector_float * foo = neurons_i2->get_state_vector("g_nmda");
		// auryn_vector_float_set_all( foo, 5.0 );
	}

	if ( eta > 0 ) {
		con_stim_e->stdp_active = true;
	} else {
		con_stim_e->stdp_active = false;
	}


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

