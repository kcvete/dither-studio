/* Inner loops for the serial dither modes (error diffusion, Riemersma).
 * Compiled by setup.sh into env/libcdither.dylib and loaded via ctypes.
 * Semantics must match static/dither.js exactly — same scan order, same
 * float32 arithmetic, same nearest-colour metric.
 * Build:  cc -O3 -ffp-contract=off -shared -fPIC -o env/libcdither.dylib cdither.c
 */
#include <stdlib.h>
#include <math.h>

/* NB: all intermediates are double, and only the working buffer is float32.
 * That is exactly what JavaScript does (Number is f64, Float32Array rounds on
 * store), and it is what keeps this file bit-identical to static/dither.js. */
static inline int nearest(const unsigned char *pal, int np, double r, double g, double b) {
    int bi = 0; double bd = 1e300;
    for (int i = 0; i < np; i++) {
        double dr = r - (double)pal[i*3], dg = g - (double)pal[i*3+1], db = b - (double)pal[i*3+2];
        double d = dr*dr + dg*dg + db*db;
        if (d < bd) { bd = d; bi = i; }
    }
    return bi;
}

/* kern is kn triples (dx, dy, weight) as floats */
void error_diffuse(float *buf, int w, int h, const unsigned char *pal, int np,
                   const float *kern, int kn, float divisor,
                   int serpentine, float strength, const float *gate) {
    for (int y = 0; y < h; y++) {
        int rev = serpentine && (y & 1);
        for (int xi = 0; xi < w; xi++) {
            int x = rev ? w - 1 - xi : xi;
            long i = (long)y * w + x, q = i * 3;
            double r = buf[q], g = buf[q+1], b = buf[q+2];
            int k = nearest(pal, np, r, g, b);
            double nr = (double)pal[k*3], ng = (double)pal[k*3+1], nb = (double)pal[k*3+2];
            buf[q] = (float)nr; buf[q+1] = (float)ng; buf[q+2] = (float)nb;
            if (gate && gate[i] <= 0.0f) continue;
            double er = (r-nr)*(double)strength/(double)divisor,
                   eg = (g-ng)*(double)strength/(double)divisor,
                   eb = (b-nb)*(double)strength/(double)divisor;
            for (int t = 0; t < kn; t++) {
                int dx = (int)kern[t*3], dy = (int)kern[t*3+1];
                float wt = kern[t*3+2];
                if (rev) dx = -dx;
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || nx >= w || ny >= h) continue;
                long nq = ((long)ny * w + nx) * 3;
                buf[nq]   = (float)((double)buf[nq]   + er*(double)wt);
                buf[nq+1] = (float)((double)buf[nq+1] + eg*(double)wt);
                buf[nq+2] = (float)((double)buf[nq+2] + eb*(double)wt);
            }
        }
    }
}

static inline void hilbert_xy(int order, long d, int *ox, int *oy) {
    long t = d; int x = 0, y = 0, rx, ry;
    for (int s = 1; s < order; s *= 2) {
        rx = 1 & (int)(t / 2);
        ry = 1 & (int)(t ^ rx);
        if (ry == 0) {
            if (rx == 1) { x = s - 1 - x; y = s - 1 - y; }
            int tmp = x; x = y; y = tmp;
        }
        x += s * rx; y += s * ry; t /= 4;
    }
    *ox = x; *oy = y;
}

void riemersma(float *buf, int w, int h, const unsigned char *pal, int np,
               int ql, float ratio, float strength, const float *gate) {
    int order = 1; while (order < w || order < h) order *= 2;
    float *wq = (float*)malloc(sizeof(float)*ql);   /* f32 store, f64 math (as JS) */
    for (int i = 0; i < ql; i++)
        wq[i] = (float)(pow((double)ratio, (double)(i+1)/(double)ql - 1.0) * (double)strength);
    float *er = (float*)calloc(ql, sizeof(float));
    float *eg = (float*)calloc(ql, sizeof(float));
    float *eb = (float*)calloc(ql, sizeof(float));
    int head = 0;
    long total = (long)order * order;
    for (long d = 0; d < total; d++) {
        int x, y; hilbert_xy(order, d, &x, &y);
        if (x >= w || y >= h) continue;
        long i = (long)y * w + x, q = i * 3;
        double ar = 0, ag = 0, ab = 0;
        for (int t = 0; t < ql; t++) {
            int j = (head + t) % ql;
            ar += (double)er[j]*(double)wq[t];
            ag += (double)eg[j]*(double)wq[t];
            ab += (double)eb[j]*(double)wq[t];
        }
        double r = (double)buf[q] + ar/(double)ql,
               g = (double)buf[q+1] + ag/(double)ql,
               b = (double)buf[q+2] + ab/(double)ql;
        int k = nearest(pal, np, r, g, b);
        double nr = (double)pal[k*3], ng = (double)pal[k*3+1], nb = (double)pal[k*3+2];
        buf[q] = (float)nr; buf[q+1] = (float)ng; buf[q+2] = (float)nb;
        int off = (gate && gate[i] <= 0.0f);
        er[head] = off ? 0.0f : (float)(r - nr);
        eg[head] = off ? 0.0f : (float)(g - ng);
        eb[head] = off ? 0.0f : (float)(b - nb);
        head = (head + 1) % ql;
    }
    free(wq); free(er); free(eg); free(eb);
}
