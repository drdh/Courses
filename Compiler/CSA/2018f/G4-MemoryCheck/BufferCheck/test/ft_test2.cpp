#include <iostream>
#include <string.h>
using namespace std;
int main()
{
  int* p1 = new int[3];
  int* p2 = new int;
  memcpy(p2, p1, 3); // warn
  return 0;
}
