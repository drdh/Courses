#!/bin/bash
nachos='../build.linux/nachos'
$nachos  -f 
$nachos  -cp h.txt h.txt 
$nachos  -cp testFileSys testFileSys
$nachos  -x testFileSys
