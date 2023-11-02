#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <sys/file.h>

static int try_acquire_lock(const char *filename) {
	int fd = open(filename, O_RDONLY | O_CREAT, 0644);
	if (fd < 0) return 0;
    
	int r = flock(fd, LOCK_EX | LOCK_NB);
	if (r < 0) {
        close(fd);
        return 0;
    }
    
    flock(fd, LOCK_UN);
    close(fd);
    
	return 1;
}

int main(int argc, char **argv) {
    if (argc - 1 != 2) {
        fprintf(stderr, "Usage: %s <L> <R>\n", argv[0]);
        return 1;
    }
    
    int L, R;
    assert(1 == sscanf(argv[1], "%d", &L));
    assert(1 == sscanf(argv[2], "%d", &R));
    assert(1 <= L && L <= R && R <= 100000000);
    
    for (int i = L; i <= R; i++) {
        static char filename[1000];
        sprintf(filename, "../run_status/lock_%d", i);
        
        // can not lock => is running
        printf("%d %d\n", i, !try_acquire_lock(filename));
    }
    
    return 0;
}
