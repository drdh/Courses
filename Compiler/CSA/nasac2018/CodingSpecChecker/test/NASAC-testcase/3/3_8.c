#include <stdio.h>
#include <stdlib.h>

/*
 * Compare two numbers, and print the result
 * Input: number1 and number2
 * Output: compare result
 */
void judge(int number1, int number2) { // no necessary return value, correct
	if (number1 == number2) 
		printf("%d is equal to %d", number1, number2);
	if (number1 < number2)
		printf("%d is smaller to %d", number1, number2);
	printf("%d is bigger to %d", number1, number2);
} 

/*
 * Compare two numbers
 * Input: number1 and number2
 * Output: compare result
 */
int main() {
	int number1, number2;
	if (scanf("%d %d\n", &number1, &number2) == 2) {
		judge(number1, number2);
	}
	else {
		printf("Input Wrong!\n");
		exit(1);
	}
	return 0;
}