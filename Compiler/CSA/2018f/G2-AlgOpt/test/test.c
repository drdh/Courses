#include <stdio.h>

int __attribute__((noinline)) fun(int x) {
  int a = (x + 1) * (x + 1) * (x + 1) * (x + 1) * (x + 1);
  int b = x * x * x * x * x + 5 * x * x * x * x + 10 * x * x * x + 10 * x * x +
          5 * x;
  int ret = a - b;
  return ret;
}

int main() {
  int sum = 0;
  int n;
  scanf("%d", &n);
  for (int i = 0; i <= n; i++) {
    sum ^= fun(i);
    sum += i;
  }
  printf("sum = %d\n", sum);
}
