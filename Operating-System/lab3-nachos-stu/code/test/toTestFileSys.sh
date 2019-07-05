#!/bin/bash
nachos='../build.linux/nachos'
$nachos -f
$nachos -cp prince.txt prince.txt
$nachos -cp testFileSys testFileSys
$nachos -x testFileSys
