/**
 * @brief JUDAH - Jacob is equipped with a text-based user interface
 *
 * @file main.hpp
 * @author  Norbert Bátfai <nbatfai@gmail.com>
 * @version 0.0.1
 *
 * @section LICENSE
 *
 * Copyright (C) 2015 Norbert Bátfai, batfai.norbert@inf.unideb.hu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @section DESCRIPTION
 *
 * JACOB, https://github.com/nbatfai/jacob
 *
 * "The son of Isaac is Jacob." The project called Jacob is an experiment
 * to replace Isaac's (GUI based) visual imagination with a character console.
 *
 * ISAAC, https://github.com/nbatfai/isaac
 *
 * "The son of Samu is Isaac." The project called Isaac is a case study
 * of using deep Q learning with neural networks for predicting the next
 * sentence of a conversation.
 *
 * SAMU, https://github.com/nbatfai/samu
 *
 * The main purpose of this project is to allow the evaluation and
 * verification of the results of the paper entitled "A disembodied
 * developmental robotic agent called Samu Bátfai". It is our hope
 * that Samu will be the ancestor of developmental robotics chatter
 * bots that will be able to chat in natural language like humans do.
 *
 */

#include <iostream>
#include <string>
#include <sstream>
#include <signal.h>
#include "samu.hpp"

Samu samu;

bool halted {false};

void save_samu ( int sig )
{

  if ( halted )
    return;
  halted = true;

#ifndef Q_LOOKUP_TABLE
  std::string samuImage {"samu.image.txt"};
  samu.save ( samuImage );
#endif

  samu.halt();
  exit ( 0 );
}

double to_samu ( int channel, SPOTriplets &tv )
{
  double r {0.0};

  try
    {
      samu.triplet ( channel, tv );
      r = samu.reward();
    }
  catch ( const char* err )
    {
      std::cerr << err << std::endl;
    }
  return r;
}


double to_samu ( int channel, std::string &msg )
{
  double r {0.0};

  try
    {
      samu.sentence ( channel, msg );
      r = samu.reward();
    }
  catch ( const char* err )
    {
      std::cerr << err << std::endl;
    }
  return r;
}

double to_samu ( int channel, std::string &msg, std::string &key )
{
  double r {0.0};

  try
    {
      samu.sentence ( channel, msg, key );
      r = samu.reward();
    }
  catch ( const char* err )
    {
      std::cerr << err << std::endl;
    }
  return r;
}

std::map<std::string, SPOTriplets> cache;

double read_cache ( std::string & key )
{
  double sum {0.0};

  for ( auto t: cache[key] )
    {
      /*
      if ( !samu.sleep() )
        break;
      */
      SPOTriplets tv;
      tv.push_back ( t );
      sum += to_samu ( 12, tv );
    }

  return sum;
}

int main ( int argc, char **argv )
{

#ifndef Q_LOOKUP_TABLE
  std::string samuImage {"samu.image.txt"};

  std::fstream samuFile ( samuImage,  std::ios_base::in );
  if ( samuFile )
    samu.load ( samuFile );
#endif

  struct sigaction sa;
  sa.sa_handler = save_samu;
  sigemptyset ( &sa.sa_mask );
  sa.sa_flags = SA_RESTART;

  sigaction ( SIGINT, &sa, NULL );
  sigaction ( SIGTERM, &sa, NULL );
  sigaction ( SIGKILL, &sa, NULL );
  sigaction ( SIGHUP, &sa, NULL );

  // Do not remove this copyright notice!
  std::cerr << "This program is Isaac, the son of Samu Bátfai."
            << std::endl
            << "Copyright (C) 2015 Norbert Bátfai"
            << std::endl
            << "License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>"
            << std::endl
            << "This is free software: you are free to change and redistribute it."
            << std::endl
            << "There is NO WARRANTY, to the extent permitted by law."
            << std::endl
            << std::endl;

  std::string test[] =
  {
    "A rare black squirrel has become a regular visitor to a suburban garden",
    "This is a car",
    "This car is mine",
    "I have a little car",
    "The sky is blue",
    "The little brown bear has eaten all of the honey",
    "I love Samu"
  };

  int j {0};
  std::string training_file = samu.get_training_file();

  for ( ; samu.run(); )
    {
      double sum {0.0};
      if ( samu.sleep() )
        {
          samu.clear_vi();
          if ( samu.get_training_file() == training_file )
            for ( int i {0}; i<7 && samu.sleep(); ++i )
              {
                sum += to_samu ( 11, test[i] );
              }
          else
            {
              std::string key = samu.get_training_file();

              if ( cache.find ( key ) == cache.end() )
                {

                  std::fstream triplet_train ( key+".triplets",  std::ios_base::in );
                  if ( triplet_train )
                    {
                      std::cerr << "triplets2cache" << std::endl;

                      while ( !triplet_train.eof() && samu.sleep() )
                        {
                          SPOTriplet t;
                          triplet_train >> t;
                          cache[key].push_back ( t );
                        }

                      triplet_train.close();

                      std::cerr << "read from cache" << std::endl;
                      auto start = std::chrono::high_resolution_clock::now();

                      sum = read_cache ( key );

                      std::cerr << std::chrono::duration_cast<std::chrono::milliseconds> (
                                  std::chrono::high_resolution_clock::now() - start ).count()
                                << " [ms]" <<  std::endl;

                    }
                  else
                    {
                      std::cerr << "read sentences from file" << std::endl;
                      auto start = std::chrono::high_resolution_clock::now();

                      std::fstream train ( samu.get_training_file(),  std::ios_base::in );
                      if ( train )
                        {
                          std::string file = key+".triplets";
                          for ( std::string line; std::getline ( train, line ) && samu.sleep(); )
                            {

#ifndef TRIPLET_CACHE
                              sum += to_samu ( 12, line );
#else
                              sum += to_samu ( 12, line, file );
#endif

                            }
                          train.close();
                        }

                      std::cerr << std::chrono::duration_cast<std::chrono::milliseconds> (
                                  std::chrono::high_resolution_clock::now() - start ).count()
                                << " [ms]" <<  std::endl;

                    }

                }
              else
                {
                  std::cerr << "read from cache" << std::endl;
                  auto start = std::chrono::high_resolution_clock::now();
		  
                  sum = read_cache ( key );
		  
                  std::cerr << std::chrono::duration_cast<std::chrono::milliseconds> (
                              std::chrono::high_resolution_clock::now() - start ).count()
                            << " [ms]" <<  std::endl;
                }

            }

          std::cerr << "###### " << ++j << "-th iter " << sum << std::endl;
        }
      else
        sleep ( 1 );
    }

  return 0;
}
