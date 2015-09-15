/**
 * @brief JUDAH - Jacob is equipped with a text-based user interface
 *
 * @file samu.cpp
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
#include "samu.hpp"

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <ncurses.h>

Disp Samu::disp;

void Samu::FamilyCaregiverShell ( void )
{
  std::string cmd_prefix = "___";

  fd_set rfds;
  struct timeval tmo;

  int sleep {0};

  if ( sleep_ )
    sleep = sleep_after_ + 1;

  for ( ; run_ ; )
    {
      
      try
        {
	  
          disp.cg_read();

          if ( ++sleep > sleep_after_ )
            {
              if ( !sleep_ )
                {
                  std::cerr << "Isaac went to sleep." << std::endl;
                  disp.log ( "Isaac went to sleep." );
                }
              sleep_ = true;
            }
          else
            {
              std::cerr << sleep << " " << std::flush;
              disp.log ( "." );
            }
        }
      catch ( std::string line )
        {

          if ( sleep_ )
            {
              std::cerr << "Isaac is awake now." << std::endl;
              disp.log ( "\nIsaac is awake now." );
            }
          sleep_ = false;
          sleep = 0;

          if ( !line.compare ( 0, cmd_prefix.length(), cmd_prefix ) )
            {
              if ( line == cmd_prefix )
                NextCaregiver();
            }
          else
            {
              try
                {
                  message ( -1, line );
                }
              catch ( const char* err )
                {
                  std::cerr << err << std::endl;
		  disp.log(err);
                }
            }
        }

      usleep ( 50*1000 );

    }

  run_ = false;
}
