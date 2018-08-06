/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <iterator>
#include <math.h>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include "particle_filter.h"

using namespace std;
static default_random_engine gen;
const double EPS = 0.000001;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  double std_x = std[0], std_y = std[1], std_theta = std[2];
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  num_particles = 100;

  // initialize particles
  for (int i = 0; i < num_particles; i++) {
    Particle P;
    P.id = i;
    P.x = x;
    P.y = y;
    P.theta = theta;
    P.weight = 1.0;

    // Sample from normal distribution around GPS
    P.x = dist_x(gen);
    P.y = dist_y(gen);
    P.theta = dist_theta(gen);

    particles.push_back(P);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  for (int i = 0; i < num_particles; i++) {
    Particle* P = &particles[i];

    // apply motion model
    if (yaw_rate == 0 || fabs(yaw_rate) < EPS) {
      P->x += velocity * delta_t * cos(P->theta);
      P->y += velocity * delta_t * sin(P->theta);
    }
    else {
      P->x += velocity / yaw_rate * (sin(P->theta + yaw_rate * delta_t) - sin(P->theta));
      P->y += velocity / yaw_rate * (cos(P->theta) - cos(P->theta + yaw_rate * delta_t));
      P->theta += yaw_rate * delta_t;
    }
    normal_distribution<double> dist_x(P->x, std_pos[0]);
    normal_distribution<double> dist_y(P->y, std_pos[1]);
    normal_distribution<double> dist_theta(P->theta, std_pos[2]);

    // sample from normal distribution to add jitter to the motion model
    P->x = dist_x(gen);
    P->y = dist_y(gen);
    P->theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  for (int obsi = 0; obsi < observations.size(); obsi++) {
    LandmarkObs* obs = &observations[obsi];
    double min_dist;
    int min_pid;

    // initiliaze with the first landmark prediction
    if (predicted.size() > 0) {
      min_dist = dist(obs->x, obs->y, predicted[0].x, predicted[0].y);
      min_pid = predicted[0].id;
    }
    else {
      min_pid = -1;
    }

    // find the closest landmark prediction to the observation
    for (int  pi = 1; pi < predicted.size(); pi++) {
      LandmarkObs* pd = &predicted[pi];
      double distance = dist(obs->x, obs->y, pd->x, pd->y);
      if (distance < min_dist) {
        min_dist = distance;
        min_pid = pd->id;
      }
    }
    obs->id = min_pid;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    const std::vector<LandmarkObs>& observations, const Map& map_landmarks)
{
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  double sum_weights = 0.0;
  double sigma_x_2 = std_landmark[0] * std_landmark[0];
  double sigma_y_2 = std_landmark[1] * std_landmark[1];

  for (int pi = 0; pi < num_particles; pi++) {
    Particle* P = &particles[pi];
    double p_x = P->x;
    double p_y = P->y;
    double p_theta = P->theta;
    vector<LandmarkObs> m_landmarks;

    // find landmarks near particle (predict the landmarks)
    for (int mi = 0; mi < map_landmarks.landmark_list.size(); mi++) {
      float landmark_x = map_landmarks.landmark_list[mi].x_f;
      float landmark_y = map_landmarks.landmark_list[mi].y_f;
      int landmark_id = map_landmarks.landmark_list[mi].id_i;
      double distance = dist(p_x, p_y, landmark_x, landmark_y);
      if ((distance < sensor_range) || (fabs(distance - sensor_range) < EPS)) {
        LandmarkObs obs;
        obs.id = landmark_id;
        obs.x = landmark_x;
        obs.y = landmark_y;
        m_landmarks.push_back(obs);
      }
    }

    // transform car measurements to car coordinates
    vector<LandmarkObs> trans_obs;
    for (int oi = 0; oi < observations.size(); oi++) {
      LandmarkObs obs;
      const LandmarkObs* meas_obs = &observations[oi];
      obs.id = meas_obs->id;
      obs.x = p_x + cos(p_theta) * meas_obs->x - sin(p_theta) * meas_obs->y;
      obs.y = p_y + sin(p_theta) * meas_obs->x + cos(p_theta) * meas_obs->y;
      trans_obs.push_back(obs);
    }

    // associate the landmarks to the car measurements
    dataAssociation(m_landmarks, trans_obs);

    // use Multivariate Gaussian Distribution to assign weights to observations
    double gauss_norm = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);

    // reset particle weight to 1.0 to get the final weight as the product of
    // all observations' probabilities
    P->weight = 1.0;
    for (int oi = 0; oi < trans_obs.size(); oi++) {
      LandmarkObs* meas_obs = &trans_obs[oi];
      LandmarkObs* pred_obs;
      int l_id = meas_obs->id;
      // find associated landmark
      for (int li = 0; li < m_landmarks.size(); li++) {
        if (l_id == m_landmarks[li].id) {
          pred_obs = &m_landmarks[li];
          break;
        }
      }

      double weight = gauss_norm * exp(-1.0 * (pow(meas_obs->x - pred_obs->x, 2.0) / (2.0 * sigma_x_2) + pow(meas_obs->y - pred_obs->y, 2.0) / (2.0 * sigma_y_2)));

      P->weight *= weight;
    }

    sum_weights += P->weight;
  }

  if (sum_weights == 0 || fabs(sum_weights) < EPS) {
    // in case sum is 0, assume particles are equally likely
    double uniform_probability = 1.0 / particles.size();
    for (vector<Particle>::iterator pi = particles.begin(); pi != particles.end(); pi++) {
      pi->weight = uniform_probability;
    }
  }
  else {
    // normalize particle weights using the sum
    for (int pi = 0; pi < particles.size(); pi++) {
      particles[pi].weight /= sum_weights;
    }
  }

}

void ParticleFilter::resample()
{
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<Particle> result;
  vector<double> weights;

  for (int pi = 0; pi < particles.size(); pi++) {
    weights.push_back(particles[pi].weight);
  }

  // first starting index of resampling wheel
  uniform_int_distribution<int> dist_ind(0, particles.size() - 1);
  int index = dist_ind(gen);

  double max_w = *max_element(weights.begin(), weights.end());

  // uniform distribution [0.0, 2.0 * max_w)
  uniform_real_distribution<double> dist_w(0.0, 2.0 * max_w);

  double beta = 0.0;

  for (int pi = 0; pi < particles.size(); pi++) {
    beta += dist_w(gen);
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % particles.size();
    }
    result.push_back(particles[index]);
  }

  particles = result;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
    const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
