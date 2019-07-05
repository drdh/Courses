const jobs = Channel{Int}(32);

const results = Channel{Tuple}(32);

function do_work()
    for job_id in jobs
        exec_time = rand()
        sleep(exec_time)                # simulates elapsed time doing actual work
                                        # typically performed externally.
        put!(results, (job_id, exec_time))
    end
end;

function make_jobs(n)
    for i in 1:n
        put!(jobs, i)
    end
end;

n = 12;

@async make_jobs(n); # feed the jobs channel with "n" jobs

@time begin
for i in 1:4 # start 4 tasks to process requests in parallel
    @async do_work()
end

while n > 0 # print out results
    job_id, exec_time = take!(results)
    println("$job_id finished in $(round(exec_time; digits=2)) seconds")
    global n = n - 1
end
end