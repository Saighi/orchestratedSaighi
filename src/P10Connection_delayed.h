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

#ifndef P10CONNECTION_DELAYED_H_
#define P10CONNECTION_DELAYED_H_

#include "auryn.h"

#define P10DECAYTERM_H 4.0
#define P10CONNECTION_DELAYED_EULERUPGRADE_STEP 0.99

namespace auryn {

class P10Connection_delayed : public DuplexConnection
{

private:
	AurynTime start_plasticity_auryn;
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


	auryn_vector_float * scaling_vector;

	AurynFloat tau_plus;
	AurynFloat tau_minus;
	AurynFloat tau_long;
	AurynFloat tau_reset;
	NeuronID * fwd_ind; 
	AurynWeight * fwd_data;

	NeuronID * bkw_ind; 
	AurynWeight ** bkw_data;

	AurynFloat beta; 
	AurynFloat beta_fudge;

	AurynWeight weight_a;
	AurynWeight weight_b;
	AurynWeight weight_c;

	AurynTime timestep_growth;

	AurynFloat tau_consolidation;
	AurynFloat delta_consolidation;
	AurynTime timestep_consolidation;
	ForwardMatrix * w_solid_matrix;


	void push_attributes();

	void init_shortcuts();

protected:

    void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
    {
        DuplexConnection::virtual_serialize(ar,version);
		// FIXME ideally the consolidation state should be stored
		// as complex matrix state from version v0.5 on.
        ar & *w_solid_matrix; 

		for (NeuronID i = 0 ; i < src->get_rank_size() ; ++i ) {
			ar & state_x->data[i];
			ar & state_u->data[i];
		}
    }

    void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
    {
        DuplexConnection::virtual_serialize(ar,version);
		// FIXME ideally the consolidation state should be stored
		// as complex matrix state from version v0.5 on.
        ar & *w_solid_matrix;

		for (NeuronID i = 0 ; i < src->get_rank_size() ; ++i ) {
			ar & state_x->data[i];
			ar & state_u->data[i];
		}
    }


public:
	AurynFloat A3_plus;
	AurynFloat A2_plus;
	AurynFloat A2_minus;
	AurynFloat delta;
	AurynFloat pot_strength;

	Trace * tr_pre;
	Trace * tr_post;
	Trace * tr_post2;
	Trace * tr_post_scaling;

	inline AurynWeight dw_pre(const NeuronID post, const AurynWeight * w);
	inline AurynWeight dw_post(const NeuronID pre, const NeuronID post, const AurynWeight * w, const AurynWeight w0);

	void propagate_forward();
	void propagate_backward();

	bool stdp_active;
	bool consolidation_active;

	P10Connection_delayed(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter=GLUT);

	P10Connection_delayed(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynFloat eta=1, 
			AurynFloat kappa=3., AurynFloat maxweight=100. , 
			AurynFloat start_plasticity = 0.,
			TransmitterType transmitter=GLUT);

	P10Connection_delayed(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05, 
			AurynFloat eta=1, 
			AurynFloat kappa=3., AurynFloat maxweight=100. , 
			AurynFloat start_plasticity = 0.,
			TransmitterType transmitter=GLUT,
			string name = "P10Connection_delayed" );

	virtual ~P10Connection_delayed();

	void init(AurynFloat eta, AurynFloat kappa, AurynFloat maxweight);
	void set_hom_trace(AurynFloat freq);
	void set_beta(AurynFloat b);
	void set_tau_d(AurynFloat taud);
  void set_tau_cons(AurynFloat taucons);
	void set_tau_f(AurynFloat tauf);
	void set_ujump(AurynFloat r);
	void set_urest(AurynFloat r);

	void consolidate();

	void set_weight_a(AurynFloat w0);
	void set_weight_c(AurynFloat w2);
	void free();

	virtual void propagate();
	virtual void evolve();

	virtual void finalize();

	void load_fragile_matrix(string filename);

	// TODO this function should be removed on the long term -- now using netstate to save
	virtual bool load_from_file(string filename);
	// TODO this function should be removed on the long term -- now using netstate to save
	virtual bool write_to_file(string filename);

	void randomize_consolidation_variables(AurynFloat mean, AurynFloat std);

};

}

#endif /*P10CONNECTION_DELAYED_H_*/
