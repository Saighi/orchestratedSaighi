#include "auryn.h"
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random.hpp>
#include <fstream>
#include <vector>


static boost::mt19937 seed(std::time(0));
static boost::variate_generator<boost::mt19937 &, boost::exponential_distribution<> > random_n(seed, boost::exponential_distribution<>());
static boost::uniform_int<> one_to_six( 1, 6 );

using namespace auryn;

void write_random_events(std::string filename, double duration, int rate,
                         double begining, double end, int nb_pattern) {

  boost::uniform_int<> one_to_six( 0, nb_pattern-1 );
  boost::variate_generator< boost::mt19937 &, boost::uniform_int<> >
                    dice(seed, one_to_six);

  AurynTime loop_grid_size = 1.0;
  if (loop_grid_size == 0)
    loop_grid_size = 1;
  AurynDouble random_time = 0.;
  AurynDouble lambda_p = 1.0 / (1.0 / rate - auryn_timestep);
  AurynTime delay = 0.;
  AurynTime begining_auryn= begining/auryn_timestep;
  AurynTime last_result = begining_auryn;
  AurynTime end_auryn= end/auryn_timestep;
  AurynTime duration_auryn = duration / auryn_timestep;
  double rd;

  if (loop_grid_size == 0)
    loop_grid_size = 1;

  std::ostringstream oss;
  oss << filename << "_readable";
  std::string filenameR = oss.str();
  std::ofstream outfile;
  outfile.open(filename.c_str(), std::ios::out);
  std::ofstream outfileR;
  outfileR.open(filenameR.c_str(), std::ios::out);
  int one_throw;
  while (last_result < end_auryn) {
    one_throw = dice();
    outfile << last_result << " " << one_throw;
    outfile << "\n";
    outfileR << last_result * auryn_timestep << " " << one_throw;
    outfileR << "\n";

    random_time = (random_n()* ((1 / lambda_p) - duration));
    delay = random_time / auryn_timestep;
    last_result = last_result + delay + duration_auryn;
    // std::cout << last_result << std::endl;

    if (last_result % loop_grid_size) { // align to temporal grid
      last_result = (last_result / loop_grid_size + 1) * loop_grid_size;
    }
  }
  outfile.close();
  outfileR.close();
}

void generate_pattern(std::string filename, AurynDouble rate,
                      int nbNeurones, AurynDouble duration) {

  AurynDouble lambda = 1.0 / (1.0 / rate - auryn_timestep);
  AurynDouble r = random_n() / lambda;
  AurynDouble x = (NeuronID)(r / auryn_timestep + 0.5);
  AurynTime auryn_duration = duration/auryn_timestep;
  AurynTime actual_time = 0;

  std::ofstream outfile;
  outfile.open(filename.c_str(), std::ios::out);

  while (actual_time < auryn_duration){
    while (x < nbNeurones) {
      outfile << actual_time*auryn_timestep<<" "<< x;
      outfile << "\n";

      AurynDouble r = random_n() / lambda;
      // we add 1.5: one to avoid two spikes per bin and 0.5 to
      // compensate for rounding effects from casting
      x += (NeuronID)(r / auryn_timestep + 1.5);
      // beware one induces systematic error that becomes substantial at high
      // rates, but keeps neuron from spiking twice per time-step
    }
    x -= nbNeurones ;
    actual_time++;
  }
  outfile.close();
}

bool contains_inf_time(AurynDouble time, std::vector<AurynTime> &time_vect){

  for (std::vector<AurynTime>::iterator it = time_vect.begin() ; it != time_vect.end(); ++it){
    if (*it < time){
      return true;
    }
  }
  return false;

}

// void generate_spiketrains(std::string filename, AurynDouble rate,
//                       int nbNeurones,  AurynDouble begining, AurynDouble end, AurynDouble deviation) {

//   AurynDouble lambda = 1.0 / (1.0 / rate - auryn_timestep);
//   AurynDouble r = random_n() / lambda;
//   AurynDouble x = 0;
//   AurynTime end_auryn= end/auryn_timestep;
//   AurynTime begining_auryn= begining/auryn_timestep;
//   AurynTime actual_time = 0;
//   std::vector<AurynTime> last_spikes (nbNeurones);

//   while(x<nbNeurones){
//     last_spikes[x]= begining_auryn + ((random_n() / lambda)/auryn_timestep);
//   }
//   x = 0;

//   std::ofstream outfile;
//   outfile.open(filename.c_str(), std::ios::out);
//   while(contains_inf_time(end_auryn,last_spikes)){
//     while (x<nbNeurones){
//       outfile << last_spikes[x]*auryn_timestep<<" "<< x;
//       outfile << "\n";
//       last_spikes[x] = last_spikes[x] + ((random_n() / lambda)/auryn_timestep);
//       x++;
//     }
//     x = 0;
//   }
//   outfile.close();
// }
