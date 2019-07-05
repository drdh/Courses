#include "../DebugInfo/Global_Data.h"
#include "../DebugInfo/Drawer.h"
#include "../DebugInfo/Sync.h"

Sync syn;
Global_data *VG = new Global_data();

void test_func1() {
    VG = syn.Load();
    VG->enter_function("1", "test_func1", 0);
    syn.Save(VG);
    int **p;
    p = (int **) malloc(10000 * sizeof(int *));
    for (int i = 0; i < 10000; i++) {
        p[i] = (int *) malloc(1000 * sizeof(int));
    }
    VG = syn.Load();
    VG->exit_function();
    syn.Save(VG);
}

int fib(int n) {
    VG = syn.Load();
    VG->enter_function("2", "fib",  0);
    syn.Save(VG);
    if (n == 0) {
        VG = syn.Load();
        VG->exit_function();
        syn.Save(VG);
        return 0;
    } else if (n == 1) {
        VG = syn.Load();
        VG->exit_function();
        syn.Save(VG);
        return 1;
    } else {
        int x = fib(n - 1) + fib(n - 2);
        VG = syn.Load();
        VG->exit_function();
        syn.Save(VG);
        return x;
    }
}

int main() {
    DrawerPick picker;
    test_func1();
    fib(3);
    DrawerBase* drawer = picker.get(Print);
    drawer->draw(*VG);
    // 这一步必须在所有的VG调用，因为会删除传入的指针
    return 0;
}
