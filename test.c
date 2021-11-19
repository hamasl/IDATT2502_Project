#include "stdio.h"

static void goodG2B() {
    char * data;
    void (*funcPtr) (char *) = goodG2BSink;
    data = NULL;
    /* FIX: Use memory allocated on the stack with ALLOCA */
    data = (char *)ALLOCA(100*sizeof(char));
    /* Initialize and make use of data */
    strcpy(data, "A String");
    printLine(data);
    funcPtr(data);
}