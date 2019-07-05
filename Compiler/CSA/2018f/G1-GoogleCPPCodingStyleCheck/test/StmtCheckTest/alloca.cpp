#include <alloca.h>
int main()
{
	int a=1;
	alloca(1);
	{{alloca(2);}}
	if(a)alloca(3);
	while(1)alloca(4*7);
	return 0;
}

int another(){
	while(1)
		for(int i = 1; i < 1; i ++){
			{
				switch(1){
					case 1: alloca(1);
				}
			}
		}
}
