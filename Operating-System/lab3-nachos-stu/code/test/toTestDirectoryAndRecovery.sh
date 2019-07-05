#!/bin/bash
nachos='../build.linux/nachos'
$nachos -f 
$nachos -cp testDirectory testDirectory -x testDirectory
#print contents
$nachos -p folder1/folder2/file
#delete file
$nachos -r folder1/folder2/file
#recover file to recovery.txt
$nachos -recover folder1/folder2/file recovery.txt