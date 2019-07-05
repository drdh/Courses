#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "Global_Data.h"
#include "Sync.h"
#include "hook.h"

Global_data *VG = new Global_data();
Sync saver;

namespace eculid {
	typedef void (*PF0)();
	 Net::Net() {
		VG = saver.Load();
		VG->enter_function("_ZN6eculid3NetC2Ev", "Net", 1);
		saver.Save(VG);
		PF0 ori = (PF0) dlsym(RTLD_NEXT, "_ZN6eculid3NetC2Ev");
		if (!ori) {
			fprintf(stderr, "%s\n", dlerror());
			exit(1);
		}
		ori();
		VG = saver.Load();
		VG->exit_function();
		saver.Save(VG);
	}


	typedef void (*PF1)();
	void Net::forward() {
		VG = saver.Load();
		VG->enter_function("_ZN6eculid3Net7forwardEv", "forward", 1);
		saver.Save(VG);
		PF1 ori = (PF1) dlsym(RTLD_NEXT, "_ZN6eculid3Net7forwardEv");
		if (!ori) {
			fprintf(stderr, "%s\n", dlerror());
			exit(1);
		}
		ori();
		VG = saver.Load();
		VG->exit_function();
		saver.Save(VG);
	}
}
