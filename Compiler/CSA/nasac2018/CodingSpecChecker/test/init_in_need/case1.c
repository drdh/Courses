extern void memset(void *, int, unsigned int);

int fun1()
{
	int array[10] = {0};
	memset(array, 0, 10 * sizeof(int));
	
	return 0;
}