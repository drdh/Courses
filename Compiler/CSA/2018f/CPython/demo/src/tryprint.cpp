#include "../DebugInfo/Global_Data.h"
#include "../DebugInfo/Drawer.h"
#include "../DebugInfo/Sync.h"

Global_data *VG = new Global_data();

int main() {
    Sync sync;
    VG = sync.Load();
    DrawerPick picker;
    DrawerBase* drawer = picker.get(Print);
    drawer->draw(*VG);
    return 0;
}
