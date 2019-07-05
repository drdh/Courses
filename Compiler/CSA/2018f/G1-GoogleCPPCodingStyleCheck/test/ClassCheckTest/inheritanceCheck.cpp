class B{
	virtual void get(){}
};
class A: private B{
	void get(){}
};
