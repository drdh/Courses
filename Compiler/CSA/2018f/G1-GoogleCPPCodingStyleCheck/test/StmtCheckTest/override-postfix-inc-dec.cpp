//#include <string>
class Calendar{
public:
	Calendar(int y = 2015, int m = 8, int d = 6, int h = 17, int mm = 9, int s = 0)
		:year(y), month(m), day(d), hour(h), minute(mm), second(s){
	}
 
	Calendar &operator++();
	Calendar operator++(int);
 
	Calendar &operator--();
	Calendar operator--(int);

	void tick(){
		second++;
	}
 
private:
	int day;
	int month;
	int year;
	int hour;
	int minute;
	int second;
};
 
Calendar& Calendar::operator++(){
	tick();
	return *this;
}
 
Calendar& Calendar::operator--(){
	tick();
	return *this;
}

Calendar Calendar::operator++(int){
	Calendar temp = *this;
	tick();
	return temp;
}

Calendar Calendar::operator--(int){
	Calendar temp = *this;
	tick();
	return temp;
}
int main(){
	Calendar a;
	a++;
	++a;
	if(1)a--;
	{{a--;}}
	--a;
	return 0;
} 
