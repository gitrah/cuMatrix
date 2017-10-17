#!/bin/bash
time make -j 12 omp=1 nvml=1 statFunc=1 cdp=1 dbg=1 && time make -j 12 omp=1 nvml=1 statFunc=1 cdp=1 dbg=1 test
