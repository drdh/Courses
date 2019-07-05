#include <stdio.h>
int main(int argc, char *argv[])
{
	// right
	int a = 0;
	int *b = NULL;
	int *c = nullptr;
	float d = 0.0;
	int A = 3.3;
	float B = 1;
	float C = (float)0;

	// error
	int e = NULL;
	float f = NULL;
	int g = 0.0;
	float h = 0;
	int *i = 0;
	int j = '\0';
	float k = '\0';
	unsigned char l = NULL;
	char m = ((((0))));
	char n = 0.0;
	char o = NULL;

	return 0;
}
