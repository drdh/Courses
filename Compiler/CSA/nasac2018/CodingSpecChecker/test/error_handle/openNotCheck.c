int cc() {return 1;}
int fopen() { return 0; }

int main()
{
  int i = 1;
  cc();
  int ret = fopen();
  cc();
}
