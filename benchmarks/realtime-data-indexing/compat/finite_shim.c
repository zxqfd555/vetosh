/* Compat shim: glibc >= 2.31 dropped the __*_finite symbols that code
 * compiled with -ffast-math (here: the vector-index math inside the dev
 * pathway engine) still references. RHEL keeps them as compat symbols, the
 * Debian-based python image does not — so the benchmark containers preload
 * this tiny library. Not needed once the engine is built by CI wheels. */
#include <math.h>

double __log_finite(double x)  { return log(x);   }
float  __logf_finite(float x)  { return logf(x);  }
double __asin_finite(double x) { return asin(x);  }
float  __asinf_finite(float x) { return asinf(x); }

typedef double v2df __attribute__((vector_size(16)));
typedef float  v4sf __attribute__((vector_size(16)));

v2df _ZGVbN2v___log_finite(v2df x) {
    v2df r; r[0] = log(x[0]); r[1] = log(x[1]); return r;
}
v4sf _ZGVbN4v___logf_finite(v4sf x) {
    v4sf r; for (int i = 0; i < 4; i++) r[i] = logf(x[i]); return r;
}
