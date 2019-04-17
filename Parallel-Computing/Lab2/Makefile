all:
	mpic++ pi.cpp -o pi
	mpiexec -n 4 ./pi
	
	mpic++ PSRS.cpp -o PSRS
	mpiexec -n 4 ./PSRS

	@#mpic++ PSRS_ref.cpp -o PSRS_ref
	@#mpiexec -n 4 ./PSRS_ref
clean:
	rm -rf pi PSRS PSRS_ref
