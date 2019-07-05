### TUCheckStateStore使用示例

```c++
#include "StateStore.h"
#include <iostream>

using namespace std;

REGISTER_MAP_WITH_TUCHECKSTATE(TestMap, int, int)

REGISTER_SET_WITH_TUCHECKSTATE(TestSet, int)

REGISTER_LIST_WITH_TUCHECKSTATE(TestList, int)

int main(int argc, char const *argv[])
{
	TUCheckStateStore &Store = TUCheckStateStore::Get();
	Store.set<TestMap>(0,1);
	Store.set<TestMap>(1,2);
	Store.contains<TestMap>(0);
	cout << *Store.get<TestMap>(0) << endl;
	cout << *Store.get<TestMap>(1) << endl;
	Store.remove<TestMap>(0);
	cout << Store.contains<TestMap>(0) << endl;

	Store.add<TestSet>(0);
	Store.add<TestSet>(0);
	Store.remove<TestSet>(0);
	cout << Store.contains<TestSet>(0) << endl;

	Store.add<TestList>(1);
	Store.add<TestList>(1);
	Store.remove<TestList>(1);
	cout << Store.contains<TestList>(1) << endl;
	Store.remove<TestList>(1);
	cout << Store.contains<TestList>(1) << endl;

	return 0;
}
```

