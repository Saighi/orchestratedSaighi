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

#include "P10Connection.h"

using namespace auryn;
using namespace std;

void P10Connection::init(AurynFloat eta, AurynFloat kappa, AurynFloat maxweight)
{
	set_name("P10Connection");

	tau_plus  = 20e-3;
	tau_minus = 20e-3;

	tau_growth = 0.25;
	timestep_growth = tau_growth/auryn_timestep;

	set_max_weight(maxweight);


	double fudge = 1;
	eta *= fudge;
	A3_plus  = 1*eta;
	A2_minus = 1*eta;

	if ( dst->get_post_size() ) {
		tr_pre = src->get_pre_trace(tau_plus);
		tr_post = dst->get_post_trace(tau_minus);

		// syn_scaling = dst->get_state_vector("syn_scaling");
		// for ( NeuronID i = 0 ; i < dst->get_post_size() ; ++i ) 
		//    auryn_vector_float_set ( syn_scaling, i, 1.0 ); 
	}


	stdp_active = true;
	constant_growth = false;
	no_triplet = false;

	// cases where dst->evolve_locally() == true will be registered in SparseConnection
	if ( src->evolve_locally() == true && dst->evolve_locally() == false )
		sys->register_connection(this);

	if ( src->get_rank_size() ) {
		// init of STP stuff
		tau_d = 0.2;
		tau_f = 1.0;
		Urest = 0.2;
		Ujump = 0.2;
		state_x = auryn_vector_float_alloc( src->get_vector_size() );
		state_u = auryn_vector_float_alloc( src->get_vector_size() );
		state_temp = auryn_vector_float_alloc( src->get_vector_size() );
		for (NeuronID i = 0; i < src->get_rank_size() ; i++)
		{
			   auryn_vector_float_set (state_x, i, 1 );
			   auryn_vector_float_set (state_u, i, Ujump );
		}

	}

	// Registering the right number of spike attributes
	add_number_of_spike_attributes(1);
}

void P10Connection::finalize() {
	DuplexConnection::finalize();
	init_shortcuts();
}

void P10Connection::init_shortcuts() 
{
	if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank

	fwd_ind = w->get_row_begin(0); 
	fwd_data = w->get_data_begin();

	bkw_ind = bkw->get_row_begin(0); 
	bkw_data = bkw->get_data_begin();
}

void P10Connection::free()
{
	if ( src->get_rank_size() > 0 ) {
		auryn_vector_float_free (state_x);
		auryn_vector_float_free (state_u);
		auryn_vector_float_free (state_temp);
	}
}

P10Connection::P10Connection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter) : DuplexConnection(source, destination, transmitter)
{

}

P10Connection::P10Connection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
		AurynFloat eta, 
		AurynFloat kappa, 
		AurynFloat maxweight , 
		AurynFloat start_plasticity,
		TransmitterType transmitter) 
: DuplexConnection(source, 
		destination, 
		filename, 
		transmitter)
{
	start_plasticity_auryn = (AurynTime) (start_plasticity/auryn_timestep);
	init(eta, kappa, maxweight);
	init_shortcuts();
}

P10Connection::P10Connection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynFloat eta, 
		AurynFloat kappa, AurynFloat maxweight , 
		AurynFloat start_plasticity,
		TransmitterType transmitter,
		string name) 
: DuplexConnection(source, 
		destination, 
		weight, 
		sparseness, 
		transmitter, 
		name)
{
	start_plasticity_auryn = (AurynTime) (start_plasticity/auryn_timestep);
	init( eta, kappa, maxweight);
	init_shortcuts();
}

P10Connection::~P10Connection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}



inline AurynWeight P10Connection::dw_pre(const NeuronID post, const AurynWeight * w)
{

	AurynDouble dw = A2_minus*(tr_post->get(post));
	
	return dw;
}

inline AurynWeight P10Connection::dw_post(const NeuronID pre, const NeuronID post, const AurynWeight * w)
{
	AurynDouble p = tr_post->get(post);

	AurynDouble dw = A3_plus*tr_pre->get(pre);

	return dw;

}

void P10Connection::propagate_forward()
{
	// loop over spikes
	for (int i = 0 ; i < src->get_spikes()->size() ; ++i ) {
		// get spike at pos i in SpikeContainer
		NeuronID spike = src->get_spikes()->at(i);

		// extract spike attribute from attribute stack;
		AurynFloat attribute = get_spike_attribute(i);

		// loop over postsynaptic targets
		for (NeuronID * c = w->get_row_begin(spike) ; 
				c != w->get_row_end(spike) ; 
				++c ) {
			AurynWeight value = fwd_data[c-fwd_ind] * attribute; 
			transmit( *c , value );
			if ( sys->get_clock()>start_plasticity_auryn ) {
			  NeuronID translated_spike = dst->global2rank(*c); // only to be used for post traces
			  fwd_data[c-fwd_ind] -= dw_pre(translated_spike,&fwd_data[c-fwd_ind]);
			  if ( fwd_data[c-fwd_ind] < 0 ) 
				fwd_data[c-fwd_ind] = 0.;
			}
		}
	}
}

void P10Connection::propagate_backward()
{
	SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
	// process spikes
	for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
			spike != spikes_end ; ++spike ) {
		NeuronID translated_spike = dst->global2rank(*spike); // only to be used for post traces
		if (sys->get_clock()>start_plasticity_auryn) {
			for (NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {

				#ifdef CODE_ACTIVATE_PREFETCHING_INTRINSICS
				_mm_prefetch(bkw_data[c-bkw_ind+2],  _MM_HINT_NTA);
				#endif

				AurynWeight * cur = bkw_data[c-bkw_ind];
				*cur = *cur + dw_post(*c,translated_spike,cur);
				if (*bkw_data[c-bkw_ind]>get_max_weight()) *bkw_data[c-bkw_ind]=get_max_weight();
				if (*bkw_data[c-bkw_ind]<get_min_weight()) *bkw_data[c-bkw_ind]=get_min_weight();
			}
		}
	}
}

void P10Connection::propagate()
{
	if ( src->evolve_locally()) {
		push_attributes(); // stuffs all attributes into the SpikeDelays for sync
	}
	if ( dst->evolve_locally() ) { // necessary 
		// remember this connection can be registered although post might be empty on this node
		propagate_forward();
		propagate_backward();
	}
}

void P10Connection::evolve()
{
	if ( src->evolve_locally() ) { 
		// dynamics of x
		auryn_vector_float_set_all( state_temp, 1);
		auryn_vector_float_saxpy(-1,state_x,state_temp);
		auryn_vector_float_saxpy(auryn_timestep/tau_d,state_temp,state_x);

		// dynamics of u
		auryn_vector_float_set_all( state_temp, Urest);
		auryn_vector_float_saxpy(-1,state_u,state_temp);
		auryn_vector_float_saxpy(auryn_timestep/tau_f,state_temp,state_u);

	}

}



void P10Connection::push_attributes()
{
	SpikeContainer * spikes = src->get_spikes_immediate();
	for (SpikeContainer::const_iterator spike = spikes->begin() ;
			spike != spikes->end() ; ++spike ) {
		// dynamics 
		NeuronID spk = src->global2rank(*spike);
		double x = auryn_vector_float_get( state_x, spk );
		double u = auryn_vector_float_get( state_u, spk );
		auryn_vector_float_set( state_u, spk, u+Ujump*(1-u) );
		auryn_vector_float_set( state_x, spk, x-u*x );

		// one attribute per spike - make sure to set set_num_spike_attributes for src
		double valueToSend = x*u;
		src->push_attribute( valueToSend ); 

		// cout.precision(5);
		// cout << " " << x << " " << u << " " << valueToSend << endl;

	}
}

void P10Connection::set_tau_f(AurynFloat tauf) {
	tau_f = tauf;
}


void P10Connection::set_tau_d(AurynFloat taud) {
	tau_d = taud;
}

void P10Connection::set_ujump(AurynFloat r) {
	Ujump = r;
}

void P10Connection::set_urest(AurynFloat r) {
	Urest = r;
}

bool P10Connection::write_to_file(string filename)
{

	std::stringstream oss;
	oss << filename << "2";

	logger->msg("Writing short-term plasticity state to file...", VERBOSE);

	oss.str("");
	oss << filename << ".cstate";
	std::ofstream outfile;
	outfile.open(oss.str().c_str(),std::ios::out);
	if (!outfile) {
		std::cerr << "Can't open output file " << filename << std::endl;
	  throw AurynOpenFileException();
	}

	boost::archive::text_oarchive oa(outfile);
	for (NeuronID i = 0 ; i < src->get_rank_size() ; ++i ) {
		oa << state_x->data[i];
		oa << state_u->data[i];
	}

	outfile.close();

	return SparseConnection::write_to_file(filename);
}


bool P10Connection::load_from_file(string filename)
{

	std::stringstream oss;
	oss << filename << "2";


	oss.str("");
	oss << filename << ".cstate";
	std::ifstream infile (oss.str().c_str());
	if (!infile) {
		std::stringstream oes;
		oes << "Can't open input file " << filename;
		logger->msg(oes.str(),ERROR);
		throw AurynOpenFileException();
	}

	boost::archive::text_iarchive ia(infile);
	for (NeuronID i = 0 ; i < src->get_rank_size() ; ++i ) {
		ia >> state_x->data[i];
		ia >> state_u->data[i];
	}

	infile.close();

	bool returnvalue = SparseConnection::load_from_file(filename);


	return returnvalue;
}

void P10Connection::load_fragile_matrix(string filename)
{
	// load fragile (w) from complete file 
	DuplexConnection::load_from_complete_file(filename);
}
