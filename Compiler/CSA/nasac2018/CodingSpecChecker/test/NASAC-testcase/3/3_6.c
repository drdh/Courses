#include <stdio.h>
#include <stdlib.h>

/*
 * Compare two numbers
 * Input: number1 and number2
 * Output: 0 means number1 is equal to number2
 *         1 means number1 is smaller than number2
 *         2 means number1 is bigger than number2
 */
int judge(int number1, int number2) {
	if (number1 == number2) 
		return 0;
	if (number1 < number2)
		return 1;
	return -1;
} 

/*
 * Compare two numbers
 * Input: number1 and number2
 * Output: compare result
 */
int main() {
	int number1, number2;
	if (scanf("%d %d\n", &number1, &number2) == 2) {
		if (judge(number1, number2) == 0) { // bad: incompletely check the return value
			printf("%d is equal to %d", number1, number2);
		}
	}
	else {
		printf("Input Wrong!\n");
		exit(1);
	}
	return 0;
}