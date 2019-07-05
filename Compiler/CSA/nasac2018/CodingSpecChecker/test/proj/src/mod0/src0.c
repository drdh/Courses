int a = 0, b = 0;

struct Vector {
    unsigned size;
    unsigned capacity;
    int *data;
};

int Init(struct Vector *V, unsigned capacity) {
    if (!V) {
        return -1;
    }
    V->data = malloc(sizeof(int) * capacity);
    if (!V->data) {
        return -1;
    }
    V->size = 0;
    V->capacity = capacity;
    return 0;
}

// int IsFull(const struct Vector *V) {
//     if (!V) {
//         return -1;
//     }
//     return V->size == V->capacity;
// }

int IsFull(const struct Vector * const V);

// Comments test
// Comments test
int Push(struct Vector *V, int E) {

    if (!V) {
        return -1;
    }

    V->data[V->size++] = E;

    return 0;
}

// int main() {
//     struct Vector V, P;
//     Init(&V, 2);
//     return P.size;
// }

// int branch(int a, int b) {
//     if (a > 10 && b < 10) {
//         return 0;
//     }
//     return 1;
// }