#include <stdlib.h>
int main(){
	int *a = malloc(0);
	*a = 0;
	return 0;
}
