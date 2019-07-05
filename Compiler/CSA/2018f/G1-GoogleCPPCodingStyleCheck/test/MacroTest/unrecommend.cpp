#define ONE 1
#define TWO "2"
#define THREE 1+2
#define FOUR -4
#define FIVE "1"+"4"
#define SIX 8-2
#define SEVEN 7/1
#define EIGHT 2*4
int i = 8;
#define NINE ++i
#define TEN i++
#define MINUS --i
#define MINUS2 i--
#define INC() i+1
#define INC2(j) j+1 
#define DO() \
do{\
}while(0)
#define MYDEBUGLOG(msg) std::cout << __FILE__ << ":" << __LINE__ << ": " << msg
#define IN(str) std::in>>str
#define foreach(list,index) for(;index < list.size; index++)


#undef ONE
#undef TWO
#undef THREE
#undef FOUR
#undef FIVE
#undef SIX
#undef SEVEN
#undef EIGHT
#undef NINE
#undef TEN

