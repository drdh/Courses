#include <string.h>
#include <stdlib.h>
int main()
{
  const char* s1 = "abc";
  char *s2 = (char *)malloc(sizeof(char));
  strcpy(s2, s1); // warn
  return 0;
}
