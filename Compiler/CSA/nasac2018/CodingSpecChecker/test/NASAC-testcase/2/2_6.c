/* Example to reverse a sentence entered by user without using strings. */

#include <stdio.h>
void reverseSentence();

int main()
{
    printf("Enter a sentence: ");
    reverseSentence();

    return 0;
}

// user-defined function to reverse a sentence
// Input: none
// Output: none
void reverseSentence()
{
    char c;
    if (scanf("%c", &c) != 1)
    {
      printf("input error.\n");
      return;
    }

    if( c != '\n')
    {
        reverseSentence();
        printf("%c",c);
    }
}
