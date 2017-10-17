# !/bin/sh
#
#		W A R N I N G   - - -  do not click outside of the terminal/konsole running gdb when 
#										  the program being debugged is in an interrupted state, or
#										  the computer will HANG and you will have to POWER OFF to continue.
#
echo "W A R N I N G   - - -  do not click outside of the terminal/konsole running gdb when \
the program being debugged (with software preemption) is in an interrupted state, or \
the computer will HANG and you will have to POWER OFF to continue."
export CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1
