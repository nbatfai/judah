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

#include <queue>
#include <cstdio>
#include <cstring>

#include "nlp.hpp"
#include "ql.hpp"

#ifndef CHARACTER_CONSOLE
#include <pngwriter.h>
#endif
#include <chrono>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <sstream>

#include "disp.hpp"

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
        throw "\nMy attention diverted elsewhere.";
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
  
class VisualImagery
{

public:

  VisualImagery ( Samu & samu ) :samu ( samu )
  {}

  ~VisualImagery()
  {}

  void operator<< ( std::vector<SPOTriplet> triplets )
  {

    if ( !triplets.size() )
      return;

    for ( auto triplet : triplets )
      {
        if ( program.size() >= stmt_max )
          program.pop();

        program.push ( triplet );
      }

    boost::posix_time::ptime now = boost::posix_time::second_clock::universal_time();
    std::string image_file = "samu_vi_"+boost::posix_time::to_simple_string ( now ) +".png";

#ifndef CHARACTER_CONSOLE
    char * image_file_p = strdup ( image_file.c_str() );
    pngwriter image ( 256, 256, 65535, image_file_p );
    free ( image_file_p );
#else
    char console[10][80];
    std::memset ( console, 0, 10*80 );
#endif

    char stmt_buffer[1024];
    char *stmt_buffer_p = stmt_buffer;

    std::queue<SPOTriplet> run = program;

#ifndef Q_LOOKUP_TABLE

    std::string prg;
    stmt_counter = 0;
    while ( !run.empty() )
      {
        auto triplet = run.front();

        prg += triplet.s.c_str();
        prg += triplet.p.c_str();
        prg += triplet.o.c_str();

        std::snprintf ( stmt_buffer, 1024, "%s.%s(%s);", triplet.s.c_str(), triplet.p.c_str(), triplet.o.c_str() );

#ifndef CHARACTER_CONSOLE
        char font[] = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf";
        char *font_p = font;

        image.plot_text_utf8 ( font_p,
                               11,
                               5,
                               256- ( ++stmt_counter ) *28,
                               0.0,
                               stmt_buffer_p, 0, 0, 0 );
#else

        std::strncpy ( console[stmt_counter++], stmt_buffer, 80 );

#endif

        run.pop();
      }

#ifndef CHARACTER_CONSOLE
    double *img_input = new double[256*256];

    for ( int i {0}; i<256; ++i )
      for ( int j {0}; j<256; ++j )
        {
          img_input[i*256+j] = image.dread ( i, j );
        }
#else
    double *img_input = new double[10*80];

    std::stringstream con;

    for ( int i {0}; i<10; ++i )
      {
        std::string ci;
        for ( int j {0}; j<80; ++j )
          {
            img_input[i*80+j] = ( ( double ) console[i][j] ) / 255.0;
            if ( isgraph ( console[i][j] ) )
              ci += console[i][j];
          }
        con << " " << i << ". " << ci << std::endl;
      }

    samu.disp.vi ( con.str() );

#endif

#else
    std::string prg;
    while ( !run.empty() )
      {
        auto triplet = run.front();

        prg += triplet.s.c_str();
        prg += triplet.p.c_str();
        prg += triplet.o.c_str();

        run.pop();
      }
#endif

    auto start = std::chrono::high_resolution_clock::now();

    std::cerr << "QL start... ";

#ifndef Q_LOOKUP_TABLE

    SPOTriplet response = ql ( triplets[0], prg, img_input );

    std::stringstream resp;

    resp << std::endl
         << samu.name
#ifdef QNN_DEBUG
         << "@"
	 << (samu.sleep_?"sleep":"awake")
         << "."	 
         << ql.get_action_count()
         << "."
         << ql.get_action_relevance()
         << "%"
#endif
         <<"> "
         << response;

    std::string r = resp.str();

    std::cerr << r << std::endl;

    samu.disp.log ( r );


#else

    std::cerr << ql ( triplets[0], prg ) << std::endl;

#endif

    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds> (
                std::chrono::high_resolution_clock::now() - start ).count()
              << " ms " <<  std::endl;

#ifndef CHARACTER_CONSOLE

#ifndef Q_LOOKUP_TABLE
    delete[] img_input;
    image.close();
#endif

#endif
  }

  double reward ( void )
  {
    return ql.reward();
  }

  void save ( std::string &fname )
  {
    ql.save ( fname );
  }

  void load ( std::fstream & file )
  {
    ql.load ( file );
  }

  void clear ( void )
  {
    while ( !program.empty() )
      {
        program.pop();
      }
  }

private:

  Samu &samu;
  QL ql;
  std::queue<SPOTriplet> program;
  int stmt_counter {0};
  static const int stmt_max = 7;

};
  
  static Disp disp;
  static std::string name;

  bool run_ {true};
  bool sleep_ {true};
  int sleep_after_ {160};
  unsigned int read_usec_{50*1000};
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread terminal_thread_ {&Samu::terminal, this};

  NLP nlp;
  VisualImagery vi{*this};

  int caregiver_idx_ {0};
  std::vector<std::string> caregiver_name_ {"Norbi", "Nandi", "Matyi", "Greta"};

  std::mutex msg_mutex;
  int old_talk_id {0};

};

#endif
