#include "Global_Data.h"
#include "Sync.h"
#include "Drawer.h"

// 以下部分供python端ctypes调用
extern "C" {
Global_data *data = new Global_data();
Sync *saver;


int save() {
    saver->Save(data);
    return 1;
}

int load() {
    data = saver->Load();
    return 1;
}

int enter_function(char *mangle, char *demangle, char language) {
    if (mangle == NULL || demangle == NULL) {
        return 0;
    }
    data->enter_function(mangle, demangle, language);
    return 1;
}

int exit_function() {
    data->exit_function();
    return 1;
}

int test(){
    DrawerPick picker;
    DrawerBase* drawer = picker.get(Print);
    drawer->draw(*data);
    return 1;
}
}

