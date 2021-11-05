/*
 * @description Infinite Loop - for()
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE835_Infinite_Loop__for_01_bad() {
  int i = 0;

  /* FLAW: Infinite Loop - for() with no break point */
  for (i = 0; i >= 0; i = (i + 1) % 256) {
    printIntLine(i);
  }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

static void good1() {
  int i = 0;

  for (i = 0; i >= 0; i = (i + 1) % 256) {
    /* FIX: Add a break point for the loop if i = 10 */
    if (i == 10) {
      break;
    }
    printIntLine(i);
  }
}
