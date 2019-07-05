typedef enum
{     
    add = 0,
    sub,
    mul,
    divi
}type;
    
int f(int (*p)(type i)){
	return 0;
}

class cl{
	public:
	int f(int (*p)(type i)){
		return 0;
	}
};

int main()
{     
    int a = 10;
    int b = 20;
      
    auto func = [=](type i)->int {
        switch (i)
        {
            case add:
                return a + b;
            case sub:
                return a - b;
            case mul:
                return a * b;
            case divi:
                return a / b;
        }
    };

	int (*p)(type i);
	cl c;

	int som = c.f(
		p = [](type i)->int {
			switch (i)
			{
				case add:
					return 0;
				case sub:
					return 1;
				case mul:
					return 2;
				default:
					return 0;
			}
		}
	);
	
	int aninteger = f(
		p = [](type i)->int {
			switch (i)
			{
				case add:
					return 0;
				case sub:
					return 1;
				case mul:
					return 2;
				default:
					return 0;
			}
		}
	 );

	auto func2 = [a, b]() {
		return a + b;
	};

	auto func4 = [&](type i)->int {
		return a + b;
	};
	auto func3 = [=, &a]() {
		return a + b;
	};
}
