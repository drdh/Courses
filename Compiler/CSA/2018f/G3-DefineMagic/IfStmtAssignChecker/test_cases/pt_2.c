int main() {
    int foo = 0;
    if (foo == 2) {
        return 0;
    }
    if (((foo < 2) && ((foo > 3) || ((foo = (5 + 2))))) == 1) {
        return -1;
    }
    return 0;
}