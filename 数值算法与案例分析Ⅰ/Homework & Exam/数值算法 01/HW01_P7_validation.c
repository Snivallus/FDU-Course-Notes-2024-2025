#include <stdio.h>
#include <float.h>

void test_float() {
    float H = 1.0f;
    int n = 2;
    while (1) {
        float term = 1.0f / (float)n;
        float newH = H + term;
        if (newH == H) break;
        H = newH;
        n++;
    }
    printf("Single precision stops at n = %d\n", n);
    printf("H_n = %.8f\n", H);
}

int main() {
    test_float();
    return 0;
}