#ifndef SAMU_HPP
#define SAMU_HPP

/**
 * @brief JUDAH - Jacob is equipped with a text-based user interface
 *
 * @file samu.hpp
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
#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>

#include "nlp.hpp"
#include "vi.hpp"



class Samu
{
public:


  
  Samu()
  {
    cv_.notify_one();
  }

  ~Samu()
  {
    run_ = false;
    terminal_thread_.join();
  }

  bool run ( void ) const
  {
    return run_;
  }

  bool halt ( void )
  {
    run_ = false;
  }


  bool sleep ( void ) const
  {
    return sleep_;
  }

  bool sleep_after ( void ) const
  {
    return sleep_after_;
  }

  void clear_vi ( void )
  {
    vi.clear();
  }

  void FamilyCaregiverShell ( void );
  void terminal ( void )
  {
    std::unique_lock<std::mutex> lk ( mutex_ );
    cv_.wait ( lk );

    FamilyCaregiverShell();
  }

  void message ( int id, std::string sentence )
  {
    if ( msg_mutex.try_lock() )
      {

        if ( id != old_talk_id )
          clear_vi();

        old_talk_id = id;

        vi << nlp.sentence2triplets ( sentence.c_str() );

        msg_mutex.unlock();

      }
    else
      {
        throw "Samu's attention diverted elsewhere.";
      }

  }

  std::string Caregiver()
  {
    if ( caregiver_name_.size() > 0 )
      return caregiver_name_[caregiver_idx_];
    else
      return "Undefined";
  }

  void NextCaregiver()
  {
    caregiver_idx_ = ( caregiver_idx_ + 1 ) % caregiver_name_.size();
  }

  double reward ( void )
  {
    return vi.reward();
  }

  void save ( std::string & fname )
  {
    vi.save ( fname );
  }

  void load ( std::fstream & file )
  {
    vi.load ( file );
  }

private:
  static Disp disp;

  bool run_ {true};
  bool sleep_ {true};
  int sleep_after_ {80};
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread terminal_thread_ {&Samu::terminal, this};

  NLP nlp;
  VisualImagery vi{&disp};

  int caregiver_idx_ {0};
  std::vector<std::string> caregiver_name_ {"Norbi", "Nandi", "Matyi", "Greta"};

  std::mutex msg_mutex;
  int old_talk_id {0};

};

#endif
