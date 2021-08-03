/* 
* Copyright 2014 Friedemann Zenke
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

#ifndef CONNECTION_BMC_H_
#define CONNECTION_BMC_H_

#include "auryn.h"


namespace auryn {

class Connection_BMC : public DuplexConnection
{

private:
	// STP parameters (maybe this should all move to a container)
	auryn_vector_float * state_x;
	auryn_vector_float * state_u;
	auryn_vector_float * state_temp;

	// Excitatory scaling
	auryn_vector_float * syn_scaling;

	double tau_d;
	double tau_f;
	double Urest;
	double Ujump;
	AurynTime start_plasticity_auryn;


	auryn_vector_float * scaling_vector;

	AurynFloat tau_plus;
	AurynFloat tau_minus;
	AurynFloat tau_reset;
	NeuronID * fwd_ind; 
	AurynWeight * fwd_data;

	NeuronID * bkw_ind; 
	AurynWeight ** bkw_data;


	AurynTime timestep_growth;


	void push_attributes();

	void init_shortcuts();



public:
	AurynFloat A3_plus;
	AurynFloat A2_minus;
	AurynFloat pot_strength;

	Trace * tr_pre;
	Trace * tr_post;
	Trace * tr_post2;
	Trace * tr_post_scaling;

	inline AurynWeight dw_pre(const NeuronID post, const AurynWeight * w);
	inline AurynWeight dw_post(const NeuronID pre, const NeuronID post, const AurynWeight * w);

	void propagate_forward();
	void propagate_backward();

	bool stdp_active;
	bool constant_growth;
	bool no_triplet;
	bool no_hetero;
	AurynFloat tau_growth;

	Connection_BMC(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter=GLUT);

	Connection_BMC(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynFloat eta=1, 
			AurynFloat kappa=3., AurynFloat maxweight=100. ,  AurynFloat start_plasticity =0,
			TransmitterType transmitter=GLUT);

	Connection_BMC(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05, 
			AurynFloat eta=1, 
			AurynFloat kappa=3., AurynFloat maxweight=100. ,  AurynFloat start_plasticity =0,
			TransmitterType transmitter=GLUT,
			string name = "Connection_BMC" );

	virtual ~Connection_BMC();

	void init(AurynFloat eta, AurynFloat kappa, AurynFloat maxweight);
	void set_hom_trace(AurynFloat freq);
	void set_tau_d(AurynFloat taud);
    void set_tau_cons(AurynFloat taucons);
	void set_tau_f(AurynFloat tauf);
	void set_ujump(AurynFloat r);
	void set_urest(AurynFloat r);

	void free();

	virtual void propagate();
	virtual void evolve();

	virtual void finalize();

	void load_fragile_matrix(string filename);

	virtual bool load_from_file(string filename);
	virtual bool write_to_file(string filename);

};

}

#endif /*CONNECTION_BMC_H_*/
