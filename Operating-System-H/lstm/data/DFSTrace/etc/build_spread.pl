#!/usr/bin/perl
#
# Build a , separated  table of summary test values
#
#   Since MS Excel limits the number of colmuns, several
# functions to print out per user data have been commented out.
#
# $Id: build_spread.pl,v 1.1 1998/11/07 22:35:02 tmkr Exp $
#                    tmk- October 98

use Getopt::Long;
use strict;
use FileHandle;
use Time::Local;

autoflush STDOUT 1;
autoflush STDERR 1;



# default  values
my $bin = "/restore/1/trace/bin";
my $output =  "-";
my $count;
my $file;
my $directory;
my @primary_user;
my $user;
my $decrease_time;
my $bogus_bytes;

my %months = (
    "Jan" => "0",
    "Feb" => "1",
    "Mar" => "2",
    "Apr" => "3",
    "May" => "4",
    "Jun" => "5",
    "Jul" => "6",
    "Aug" => "7",
    "Sep" => "8",
    "Oct" => "9",
    "Nov" => "10",
    "Dec" => "11"
    );

my  @operations =  qw(OPEN CLOSE STAT LSTAT SEEK EXECVE EXIT FORK CHDIR 
		    UNLINK ACCESS READLINK CREAT CHMOD SETREUID RENAME 
		    RMDIR LINK CHOWN MKDIR SYMLINK SETTIMEOFDAY MOUNT 
		    UNMOUNT TRUNCATE CHROOT MKNOD UTIMES READ WRITE
		    LOOKUP GETSYMLINK ROOT  );

my $no_file_Op = "SETTIMEOFDAY EXIT FORK SETREUID SETTIMEOFDAY UTIMES";

umask 22;

#parse options
&GetOptions('bin=s',\$bin,'output=s',\$output)  ||  &display_usage; 
if ($#ARGV <0 )  {     
    &display_usage;
}
#print the command line options
print "Building summary spread sheet:  output: $output  sources: @ARGV\n";
 
#$out =STDOUT;
if (-f  "$output") {
    print "File $output exists appending data\n";
    open OUTPUT, ">>$output" || die "Can't open output file $output";
}
else {
    open OUTPUT, ">$output" || die "Can't open output file $output";
}

print_header(); 

foreach $file (@ARGV) {
    print "Argument $file\n";

    
	&build_spread (glob "$file/*");

#	&build_spread ($file);
}

print "\nDone\n";

exit(0);

#Run the tests on  the set of files passed to the subroutine
#
sub  build_spread {
    foreach $directory  (@_){
	print "$directory\n";
	$directory =~ /.+\/(.+)\.(\w+\.cs\.cmu\.edu)\.(9\d)\.(\d\d)\.(\d\d).(\d\d)\:(\d\d)/ || die "ERROR- Unable to parse dir $directory";
	
	print OUTPUT "$directory.gz, $1, $2, $3,$4,$5,$6,$7,  ";
	#parse tstat
	if (! open INPUT, "$directory/tstat") {
	    print "ERROR-Unable to find tstat file for $directory\n";
	    next;
	}
#	print OUTPUT "  tstat,"; --- done inside the sub
	&process_stat;
	close INPUT;

	# process patterns	
	if (! open INPUT, "$directory/patterns") {
	    print "ERROR- Unable to find patterns file for $directory\n";
	    next;
	}
	print OUTPUT "   patterns,";
	&process_patterns;
	close INPUT;

	# process users	
	if (! open INPUT, "$directory/users") {
	    print "ERROR- Unable to find  users file for $directory\n";
	    next;
	}
	print OUTPUT "  users,";
	&process_users;
	if (defined @primary_user) {
	    print OUTPUT "@primary_user,  ";
	}
	else {
	    print OUTPUT "no-user,   ";
	}
	close INPUT;

# 
#  	foreach $user ( @primary_user) {
#  	    if (! open INPUT, "$directory/tstat.$user") {
#  		print "ERROR-Unable to find tstat.$user file for $directory\n";
#  		next;
#  	    }
#  	    print OUTPUT "  tstat.$user,";
#  	    &process_user_stat;
#  	    close INPUT;

#  	    # process patterns	
#  	    if (! open INPUT, "$directory/patterns.$user") {
#  		print "ERROR- Unable to find patterns.$user file for $directory\n";
#  		next;
#  	    }
#  	    print OUTPUT "   patterns.$user,";
#  	    &process_patterns;
#  	}

	print OUTPUT "\n";
    }
}


sub process_pattern {
    my $whole_access;
    my $whole_bytes;

    $_=<INPUT> || die "ERROR--Parsing patterns $_";
    $_=<INPUT> || die "ERROR--Parsing  patterns $_"; 
    /Whole\-file\s+(\d+)\s+\(\s*\d+\.\d\)\s+(\d+)/  || die "ERROR--Parsing patterns $_";
    $whole_access = $1;
    $whole_bytes = $2;
    $_=<INPUT> || die "ERROR--Parsing  patterns $_";
    /$_[0]\s+(\d+)\s+\(\s*\d+\.\d\)\s+(\d+)\s+\(\s*\d+\.\d\)\s+Other Seq\s+(\d+)\s+\(\s*\d+\.\d\)\s+(\d+)/ || die "ERROR--Parsing patterns $_";
    print  OUTPUT "$1,$2,$whole_access, $whole_bytes, $3,$4,";
    $_=<INPUT> || die "ERROR--Parsing  patterns $_";   
    /^\s+Random\s+(\d+)\s+\(\s*\d+\.\d\)\s+(\d+)/  || die "ERROR--Parsing patterns $_";
    print  OUTPUT "$1,$2,";

}

sub process_patterns {
    while (<INPUT>) {
	if (/^Access Type\s+Accesses/) {
	    last;
	}	
    }
    process_pattern ("Read-only");
    process_pattern ("Write-only");
    process_pattern ("Read-write");
    $_=<INPUT> || die "ERROR--Parsing  patterns $_"; 
    /Total\s+(\d+)\s+(\d+)/  || die "ERROR--Parsing patterns $_";
    print OUTPUT "$1, $2,";
}

sub process_users {
    # skip warnings
    while (<INPUT>) {
	if (/^uid\s+processes\s+records/) {
	    last;
	}	
    }
    $count = 0;
    undef @primary_user;
    while (<INPUT>) {
	/^(\w+)\s+(\d+)\s+(\d+)/ || die "ERROR--Parsing  users $_";
	print OUTPUT "$1,$2,$3,  ";
	if ($1 > 99) {
	    push  @primary_user, $1;
	}
	$count++;
	if ($count== 10) {
	    return;
	}
    }
    for (;$count < 10; $count++) {
	print OUTPUT "no-user, 0,0, ";
    }
}


sub process_stat {
    # skip warnings
    $decrease_time=0;
    $bogus_bytes = 0;
    while (<INPUT>) {
	if (/^Trace of host /) {
	    last;
	}
	if (/^Warning: Decreasing timestamp./) {
	    $decrease_time++;
	}
	if (/^Bogus bytes! chunk/) {
	    $bogus_bytes++;
	}
    }

    print OUTPUT "$decrease_time, $bogus_bytes, tstat ";
    if (! /^Trace of host (128\.2\.\d+\.\d+), /) {
	print "ERROR--Unable to parse tstat $_\n";
	return;
    }
    print OUTPUT "$1,";
    if ( / versions (\d.\d, \d.\d, \d.\d)/)  {
	print OUTPUT "$1, ";

	$_=<INPUT>|| die "ERROR--Parsing tstat $_";
	/Host booted (.+), agent started/ || die "ERROR--Parsing tstat $_";
	print OUTPUT "$1,";
    }
    else {
	print OUTPUT "1.0, 1.0, 1.0, ";
	/booted at (.+)$/ || die "ERROR--Parsing tstat $_";
	print OUTPUT "$1,";
    }
    $_=<INPUT> || die "ERROR--Parsing tstat $_";
    /Trace starts (.+), ends (.+)$/|| die "ERROR--Parsing tstat $_";
    my $start =$1;
    my $end = $2;
    print OUTPUT "$start,$end,";
    $start =~/\w\w\w (\w\w\w)\s+(\d+) (\d\d):(\d\d):(\d\d) (\d\d\d\d)/|| die "ERROR--Parsing start date $_ $start";
    my $month = $1;    my $day = $2;     my $hour = $3;
    my $minute = $4;   my $seconds = $5; my $year=$6;
    my $s =timelocal( $seconds, $minute, $hour, $day, $months{$month}, $year);
 #   print "\n\n s: $start $month $day $s ==".localtime($s)."\n";
  
    $end =~/\w\w\w (\w\w\w)\s+(\d+) (\d\d):(\d\d):(\d\d) (\d\d\d\d)/ || die "ERROR--Parsing end date $_ $end";
    $month = $1;   $day = $2;     $hour = $3;
    $minute = $4;  $seconds = $5; $year=$6;
    my $e =timelocal( $seconds, $minute, $hour, $day, $months{$month}, $year);

    my $len = ($e-$s)/3600;

#    print "\n\n s: $end $month $day $e $len  ==".localtime($e)."\n";
    printf OUTPUT "%10.4f,", $len;
    $_=<INPUT> || die "ERROR--Parsing tstat $_";
    /^(\d+) bytes, (\d+) raw records \(\s+(\d+\.\d+) \/sec\), (\d+) records,/
	|| die "ERROR--Parsing tstat $_";
    print OUTPUT "$1,$2,$3 ,$4 ,";
    $_=<INPUT> || die "ERROR--Parsing tstat $_";
    $_=<INPUT> || die "ERROR--Parsing tstat $_";
    while (<INPUT>) {
	/^\s*([A-Z]+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)/
	    || die "ERROR--Parsing tstat $_";
	my $operation = $1;
	my $count = $2;
	if ( $no_file_Op =~/$operation/ ) {
	    print OUTPUT "$count,  ";
	    next;
	}
	print OUTPUT "$2,$4,$5,$6,$7,$8,  ";
    }
}

sub process_user_stat {
    # skip warnings
    while (<INPUT>) {
	if (/^Trace of host /) {
	    last;
	}
    }
    if (! /^Trace of host (128\.2\.\d+\.\d+), /) {
	print "ERROR--Unable to parse tstat $_\n";
	return;
    }
    print OUTPUT "$1,";
    if ( / versions (\d.\d, \d.\d, \d.\d)/)  {

	$_=<INPUT>|| die "ERROR--Parsing tstat $_";
	/Host booted (.+), agent started/ || die "ERROR--Parsing tstat $_";

    }
    else {
	/booted at (.+)$/ || die "ERROR--Parsing tstat $_";
    }
    $_=<INPUT> || die "ERROR--Parsing tstat $_";
    /Trace starts (.+), ends (.+)$/ || die "ERROR--Parsing tstat $_";

    $_=<INPUT> || die "ERROR--Parsing tstat $_";
    /^(\d+) bytes, (\d+) raw records \(\s+(\d+\.\d+) \/sec\), (\d+) records,/
	|| die "ERROR--Parsing tstat $_";
    print OUTPUT "$1,$2,$3,$4,     ";
    $_=<INPUT> || die "ERROR--Parsing tstat $_";
    $_=<INPUT> || die "ERROR--Parsing tstat $_";
    while (<INPUT>) {
	/^\s*([A-Z]+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)/
	    || die "ERROR--Parsing tstat $_";
	print OUTPUT "$2,$4,$5,$6,$7,$8,  ";
    }
}

sub print_header {
    my $Op;
    print OUTPUT "File name, host, domain, year, month, day, hour, minutes, decreasing time, bogus bytes";
    print OUTPUT " marker, IP address, kern. trace vers., agent vers, coll. vers., boot time, start time, stop time, hours, total bytes, # raw records, ops/sec, # records,";
    foreach $Op (@operations) {
	if ( $no_file_Op =~/$Op/ ) {
	    print OUTPUT " $Op #,";
		next;
	}
	print OUTPUT " $Op #, fail,  ufs, afs, cfs, nfs,  ";
	}
    print OUTPUT "marker,RO accesses, bytes, # whole-file, bytes, # sequential, bytes, # random, bytes," ;
    print OUTPUT "WO accesses, bytes, # whole-file, bytes, # sequential, bytes,# random, bytes," ;
    print OUTPUT "RW accesses, bytes, # whole-file, bytes,# sequential, bytes, # random, bytes, total accesses, total bytes," ;
    print  OUTPUT " marker,";
    for (my $index= 0;$index < 20; $index++) {
	print  OUTPUT " UID, # proc, # recs,";
    }
    print OUTPUT "\n";
}
sub display_usage {
    print "$0 [-output output_file] source [source...]\n";
    exit(1);
}

