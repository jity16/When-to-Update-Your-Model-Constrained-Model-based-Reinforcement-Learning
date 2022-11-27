#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <sys/file.h>

static const char * cmd_template = "cd cat-mbrl-lib/ && "
    "CUDA_VISIBLE_DEVICES=%s "  // gpu_id
    "des=__catrunner_%s__ "  // run_id
    "algo=%s " //algo_name
    "freq=custom__%s "  // freq_name
    "seed_override=%s "  // seed
    "python -m mbrl.examples.main "
    "algorithm=mbpo "
    "overrides=%s "  // env_name
    "> ../run_status/stdout_%s.txt "  // run_id
    "2> ../run_status/stderr_%s.txt "  // run_id
;

static int acquire_lock(const char *filename, int *p_fd) {
	*p_fd = open(filename, O_RDONLY | O_CREAT, 0644);
	if (*p_fd < 0) return 0;
	int r = flock(*p_fd, LOCK_EX);
	if (r < 0) return 0;
	return 1;
}

int main(int argc, char **argv) {
    if (argc - 1 != 6) {
        fprintf(stderr, "Usage: %s <gpu_id> <run_id> <algo_name> <freq_name> <seed> <env_name>\n", argv[0]);
        return 1;
    }
    
    const char *gpu_id = argv[1];
    const char *run_id = argv[2];
    const char *algo_name = argv[3];
    const char *freq_name = argv[4];
    const char *seed = argv[5];
    const char *env_name = argv[6];
    
    
    static char cmd[1000];
    sprintf(
        cmd, cmd_template,
        gpu_id, run_id, algo_name, freq_name, seed, env_name, 
        run_id, run_id
    );
    
    static char lock_filename[1000];
    sprintf(lock_filename, "run_status/lock_%s", run_id);
    int fd;
    assert(1 == acquire_lock(lock_filename, &fd));
    
    int res = system(cmd);
    
    if (res != 0) {
        sprintf(cmd, "touch run_status/has_error_%s", run_id);
        assert(0 == system(cmd));
    }
    
    flock(fd, LOCK_UN);
    close(fd);
    
    return res;
}
