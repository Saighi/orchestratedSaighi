#include <string>
#include <sstream>
#include <fstream>
#include "auryn.h"
#include "generate_events.cpp"
#include "P10Connection.h"

using namespace auryn;

int main(int ac, char *av[])
{

    double time_simulation = 1.0; 
    //int Exc_input_neurons = 202;
    int Stim_input_neurons = 180;
    //int Inh_input_neurons = 50;
    float eta = 1e-3;
    int kappa = 10;
    float wmax = 5.0;
    int stim_rate = 2;
    int pattern_rate = 4;
    double time_pattern=0.1;
    int nb_pattern = 2;
    char strbuf [255]; 

 
    write_random_events("../../data/sim_neuron/last_sim/pattern_time",time_pattern, pattern_rate,
                         0, time_simulation,nb_pattern);


    for (int i = 0; i < nb_pattern; i++) {
        std::string file_name = "../../data/sim_neuron/last_sim/pattern";
        sprintf(strbuf, "%s_%i",file_name.c_str(),i);
        generate_pattern(string(strbuf),
                        stim_rate, Stim_input_neurons, time_pattern); 
    }

    auryn_init(ac, av);

    // PoissonFileInputGroup *stim = new PoissonFileInputGroup(Stim_input_neurons, var, true, 0.1, stim_rate);
    PoissonFileInputGroupRandom *stim = new PoissonFileInputGroupRandom(Stim_input_neurons, "../../data/sim_neuron/last_sim/pattern","../../data/sim_neuron/last_sim/pattern_time",0,nb_pattern); //CHANGE

    AIF2Group *neuron = new AIF2Group(1);
    //neuron->dg_adapt2 = 0;//make changed !!
    //neuron->dg_adapt1 = 0; // make changed !!
    // neuron->set_tau_gaba(10e-3);
    // neuron->set_tau_nmda(100e-3);
    // neuron->set_ampa_nmda_ratio(0.2);

    float weight = 0.5; // conductance amplitude in units of leak conductance
    P10Connection *conStim = new P10Connection(stim, neuron, weight, 1, eta,
                                               kappa,
                                               wmax);

    conStim -> consolidation_active = false;
    //conStim -> set_tau_cons(100);

    // AllToAllConnection * conExtExc = new AllToAllConnection(extExc,
    // neuron); AllToAllConnection * conInh = new AllToAllConnection(inh,
    // neuron);
    conStim->set_transmitter(GLUT);


    SpikeMonitor *input_spike_mon = new SpikeMonitor(stim, sys->fn("../../data/sim_neuron/last_sim/input", "ras"));
    SpikeMonitor *output_spike_mon = new SpikeMonitor(neuron, sys->fn("../../data/sim_neuron/last_sim/output", "ras"));
    WeightMonitor *wmon = new WeightMonitor(conStim, sys->fn("../../data/sim_neuron/last_sim/output", "weight"), 1.0);
    wmon->add_equally_spaced(50);

    VoltageMonitor * output_voltage_mon = new VoltageMonitor( neuron, 0, sys->fn("../../data/sim_neuron/last_sim/output","mem"),0.001 );
    PopulationRateMonitor *pmon = new PopulationRateMonitor(neuron, sys->fn("../../data/sim_neuron/last_sim/output", "prate"), 0.01);
    PopulationRateMonitor *pmonExt = new PopulationRateMonitor(stim, sys->fn("../../data/sim_neuron/last_sim/outputExt", "prate"), 0.001);
    StateMonitor *nmdaMon = new StateMonitor(neuron, 0, "g_nmda", sys->fn("../../data/sim_neuron/last_sim/g_nmda", "state"), 0.001);
    StateMonitor *adapt2mon = new StateMonitor(neuron, 0, "g_adapt2", sys->fn("../../data/sim_neuron/last_sim/g_adapt2", "state"), 0.001);
    StateMonitor *adapt1mon = new StateMonitor(neuron, 0, "g_adapt1", sys->fn("../../data/sim_neuron/last_sim/g_adapt1", "state"), 0.001);
    StateMonitor *ampamon = new StateMonitor(neuron, 0, "g_ampa", sys->fn("../../data/sim_neuron/last_sim/g_ampa", "state"), 0.001);
    StateMonitor *patmon = new StateMonitor(neuron, 0, "in_pattern",sys->fn("../../data/sim_neuron/last_sim/in_pattern", "state"), 0.001);


    sys->run(time_simulation);

    auryn_free();
}
