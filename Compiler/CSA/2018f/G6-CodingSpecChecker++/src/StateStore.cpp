#include "StateStore.h"

TUCheckStateStore::~TUCheckStateStore() {
	for (auto Pair : GDM) {
		delete Pair.second;
	}
}

TUCheckStateStore& TUCheckStateStore::Get() {
	static TUCheckStateStore Store;
	return Store;
}