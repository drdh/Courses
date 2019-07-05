#include "quantum.h"

int main() {
    Level tower[N];
    int total_level = translate(tower);
    printf("\n重新打印代码:\n");
    for (int i = 1, line = 1; i <= total_level; i++) {
        auto p = tower[i].head->next;
        while (p) {
            printf("%d: ", line);
            line++;
            p->printCode();
            printf("\n");
            p = p->next;
        }
    }
}