// cfna_engine.cpp — dependency-free C++17 inference engine for CFNA.
//
// Faithful port of the microtorch reference (cfna/microtorch/): every formula
// below is transcribed from nn.py / functional.py / cfna_blocks.py /
// cfna_model.py, float64 throughout so outputs are parity-testable against
// the NumPy oracle at ~1e-9 (tests/test_cpp_engine.py). Design notes,
// weight-order contract, and honest limitations: cpp/README.md.
//
// Two execution paths:
//   --logits  : dense full forward (mirrors MicroCFNAModel.forward), prints
//               the last position's 256 logits — the parity interface.
//   default   : incremental generation (port of cfna/microtorch/incremental
//               .py — unit-stack cache + windowed perception/decoder), the
//               fast path used for actual text generation.
//
// Build:  make -C cpp        Run:  ./cpp/cfna_run model.bin --prompt "Hi"
//
// Not implemented (documented, not silently wrong): retrieval context,
// batching, sampling parity with NumPy's RNG (greedy is byte-exact; sampled
// text uses std::mt19937 and is distribution-equal, stream-different),
// float32/int8 weights.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using std::size_t;
using std::vector;

// ----------------------------------------------------------------- tensors
struct Mat {                      // row-major [rows, cols]
    int r = 0, c = 0;
    vector<double> v;
    Mat() = default;
    Mat(int r_, int c_) : r(r_), c(c_), v((size_t)r_ * c_, 0.0) {}
    double* row(int i) { return v.data() + (size_t)i * c; }
    const double* row(int i) const { return v.data() + (size_t)i * c; }
};

// y[T,out] = x[T,in] @ W[out,in]^T (+ b[out])
static Mat linear(const Mat& x, const Mat& W, const vector<double>* b) {
    Mat y(x.r, W.r);
    for (int t = 0; t < x.r; ++t) {
        const double* xt = x.row(t);
        double* yt = y.row(t);
        for (int o = 0; o < W.r; ++o) {
            const double* w = W.row(o);
            double s = b ? (*b)[o] : 0.0;
            for (int i = 0; i < x.c; ++i) s += xt[i] * w[i];
            yt[o] = s;
        }
    }
    return y;
}

static inline double sigmoid_(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static inline double silu_(double x)    { return x * sigmoid_(x); }
static inline double gelu_(double x) {   // tanh approximation, as the oracle
    static const double K = std::sqrt(2.0 / M_PI);
    return 0.5 * x * (1.0 + std::tanh(K * (x + 0.044715 * x * x * x)));
}
static inline double softplus_(double x) { return std::log(std::exp(x) + 1.0); }

static void rmsnorm(Mat& x, const vector<double>& gain) {   // in place, eps 1e-6
    for (int t = 0; t < x.r; ++t) {
        double* xt = x.row(t);
        double ms = 0.0;
        for (int i = 0; i < x.c; ++i) ms += xt[i] * xt[i];
        double inv = std::pow(ms / x.c + 1e-6, -0.5);
        for (int i = 0; i < x.c; ++i) xt[i] *= inv * gain[i];
    }
}

// Oracle masked_softmax: e = exp(s + (mask?0:-1e30)) * mask; e / (sum + 1e-30).
// Deliberately NO max-shift (functional.py applies none here).
static void masked_softmax_row(double* s, const uint8_t* mask, int n) {
    double denom = 0.0;
    for (int j = 0; j < n; ++j) {
        s[j] = mask[j] ? std::exp(s[j]) : 0.0;
        denom += s[j];
    }
    denom += 1e-30;
    for (int j = 0; j < n; ++j) s[j] /= denom;
}

// ------------------------------------------------------------------ loader
struct Reader {
    std::ifstream f;
    explicit Reader(const std::string& p) : f(p, std::ios::binary) {}
    template <class T> T get() { T x; f.read((char*)&x, sizeof(T)); return x; }
    void tensor(vector<int>& shape, vector<double>& out) {
        uint32_t nd = get<uint32_t>();
        shape.resize(nd);
        size_t n = 1;
        for (uint32_t i = 0; i < nd; ++i) { shape[i] = (int)get<uint32_t>(); n *= shape[i]; }
        out.resize(n);
        f.read((char*)out.data(), (std::streamsize)(n * 8));
    }
    Mat mat(int er = -1, int ec = -1) {           // 2-D tensor
        vector<int> s; Mat m;
        tensor(s, m.v);
        assert(s.size() == 2);
        m.r = s[0]; m.c = s[1];
        if (er >= 0) assert(m.r == er && m.c == ec);
        return m;
    }
    vector<double> vec(int en = -1) {             // 1-D tensor
        vector<int> s; vector<double> v;
        tensor(s, v);
        assert(s.size() == 1);
        if (en >= 0) assert((int)v.size() == en);
        return v;
    }
    vector<double> raw(vector<int>& s) { vector<double> v; tensor(s, v); return v; }
};

struct Config {
    int byte_embed_dim, d_local, d_model, p_max, physical_blocks, logical_depth,
        n_heads, unit_window, decoder_window, decoder_layers, d_state,
        channel_dim, ret_byte_dim, min_patch, max_patch;
    double tau;
    bool trainable_segmentation;
};

// ------------------------------------------------------------- attention
struct Attn {                                      // shared by MHA/Sparse/Cross
    Mat q, k, v, o;                                // [d,d] (k,v may be [d,ctx])
    int h, hd;
    void load(Reader& r, int d, int nh) { q = r.mat(); k = r.mat(); v = r.mat(); o = r.mat(); h = nh; hd = d / nh; }
};

// Causal windowed self-attention (window<0 = unbounded), optional key padding.
static Mat self_attention(const Attn& A, const Mat& x, int window,
                          const vector<uint8_t>* kp, int topk /*-1 = off*/) {
    int T = x.r, d = x.c, H = A.h, hd = A.hd;
    Mat Q = linear(x, A.q, nullptr), K = linear(x, A.k, nullptr), V = linear(x, A.v, nullptr);
    Mat out(T, d);
    double scale = 1.0 / std::sqrt((double)hd);
    vector<double> sc(T);
    vector<uint8_t> mk(T);
    vector<double> masked(T);
    for (int hh = 0; hh < H; ++hh) {
        int off = hh * hd;
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < T; ++j) {
                bool m = j <= t && (window < 0 || (t - j) < window);
                if (kp && !(*kp)[j]) m = false;
                mk[j] = m ? 1 : 0;
                double s = 0.0;
                const double* qt = Q.row(t) + off;
                const double* kj = K.row(j) + off;
                for (int i = 0; i < hd; ++i) s += qt[i] * kj[i];
                sc[j] = s * scale;
            }
            if (topk > 0 && topk < T) {            // oracle's tie-keeping top-k
                for (int j = 0; j < T; ++j) masked[j] = mk[j] ? sc[j] : -1e30;
                vector<double> srt(masked);
                std::nth_element(srt.begin(), srt.end() - topk, srt.end());
                double kth = srt[srt.size() - topk];
                for (int j = 0; j < T; ++j) if (masked[j] < kth) mk[j] = 0;
            }
            masked_softmax_row(sc.data(), mk.data(), T);
            double* ot = out.row(t) + off;
            for (int j = 0; j < T; ++j) {
                if (sc[j] == 0.0) continue;
                const double* vj = V.row(j) + off;
                for (int i = 0; i < hd; ++i) ot[i] += sc[j] * vj[i];
            }
        }
    }
    return linear(out, A.o, nullptr);
}

// Cross-attention x[Tq,d] -> ctx[Tc,d] with boolean mask [Tq,Tc].
static Mat cross_attention(const Attn& A, const Mat& x, const Mat& ctx,
                           const vector<uint8_t>& mask) {
    int Tq = x.r, Tc = ctx.r, d = x.c, H = A.h, hd = A.hd;
    Mat Q = linear(x, A.q, nullptr), K = linear(ctx, A.k, nullptr), V = linear(ctx, A.v, nullptr);
    Mat out(Tq, d);
    double scale = 1.0 / std::sqrt((double)hd);
    vector<double> sc(Tc);
    for (int hh = 0; hh < H; ++hh) {
        int off = hh * hd;
        for (int t = 0; t < Tq; ++t) {
            const uint8_t* mrow = mask.data() + (size_t)t * Tc;
            for (int j = 0; j < Tc; ++j) {
                double s = 0.0;
                const double* qt = Q.row(t) + off;
                const double* kj = K.row(j) + off;
                for (int i = 0; i < hd; ++i) s += qt[i] * kj[i];
                sc[j] = s * scale;
            }
            masked_softmax_row(sc.data(), mrow, Tc);
            double* ot = out.row(t) + off;
            for (int j = 0; j < Tc; ++j) {
                if (sc[j] == 0.0) continue;
                const double* vj = V.row(j) + off;
                for (int i = 0; i < hd; ++i) ot[i] += sc[j] * vj[i];
            }
        }
    }
    return linear(out, A.o, nullptr);
}

// ----------------------------------------------------------------- modules
struct GatedMLP { Mat up, gate, down;
    void load(Reader& r) { up = r.mat(); gate = r.mat(); down = r.mat(); }
    Mat fwd(const Mat& x) const {
        Mat u = linear(x, up, nullptr), g = linear(x, gate, nullptr);
        for (size_t i = 0; i < u.v.size(); ++i) u.v[i] *= silu_(g.v[i]);
        return linear(u, down, nullptr);
    }
};

struct MLP2 { Mat w1, w2; vector<double> b1, b2;   // oracle MLP: fc2(gelu(fc1))
    void load(Reader& r) { w1 = r.mat(); b1 = r.vec(); w2 = r.mat(); b2 = r.vec(); }
    Mat fwd(const Mat& x) const {
        Mat h = linear(x, w1, &b1);
        for (double& t : h.v) t = gelu_(t);
        return linear(h, w2, &b2);
    }
};

struct SSM {                                       // CFNASelectiveSSM
    int d_inner, n, dt_rank, conv_k = 4;
    Mat in_proj, x_proj, dt_proj_w, A_log, out_proj;
    vector<double> conv_w, conv_b, dt_proj_b, D;   // conv_w flat [d_inner,1,K]
    void load(Reader& r, int d_model, int d_state) {
        d_inner = 2 * d_model; n = d_state; dt_rank = std::max(1, d_model / 16);
        in_proj = r.mat(2 * d_inner, d_model);
        vector<int> s; conv_w = r.raw(s); assert(s[0] == d_inner && s[2] == conv_k);
        conv_b = r.vec(d_inner);
        x_proj = r.mat(dt_rank + 2 * n, d_inner);
        dt_proj_w = r.mat(d_inner, dt_rank); dt_proj_b = r.vec(d_inner);
        A_log = r.mat(d_inner, n);
        D = r.vec(d_inner);
        out_proj = r.mat(d_model, d_inner);
    }
    Mat fwd(const Mat& x, const vector<uint8_t>* kp) const {
        int T = x.r;
        Mat xz = linear(x, in_proj, nullptr);              // [T, 2*di]
        Mat xin(T, d_inner), z(T, d_inner);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < d_inner; ++i) { xin.row(t)[i] = xz.row(t)[i]; z.row(t)[i] = xz.row(t)[d_inner + i]; }
        // depthwise causal conv (K=4) + silu, channel-major access
        Mat xc(T, d_inner);
        for (int c = 0; c < d_inner; ++c) {
            const double* w = conv_w.data() + (size_t)c * conv_k;
            for (int t = 0; t < T; ++t) {
                double s = conv_b[c];
                for (int i = 0; i < conv_k; ++i) {
                    int src = t - (conv_k - 1 - i);
                    if (src >= 0) s += w[i] * xin.row(src)[c];
                }
                xc.row(t)[c] = s;
            }
        }
        for (double& t : xc.v) t = silu_(t);
        Mat dbc = linear(xc, x_proj, nullptr);             // [T, dtr+2n]
        Mat delta_in(T, dt_rank);
        for (int t = 0; t < T; ++t) std::memcpy(delta_in.row(t), dbc.row(t), dt_rank * 8);
        Mat delta = linear(delta_in, dt_proj_w, &dt_proj_b);
        for (double& t : delta.v) t = softplus_(t);
        if (kp) for (int t = 0; t < T; ++t) if (!(*kp)[t]) std::memset(delta.row(t), 0, (size_t)d_inner * 8);
        vector<double> h((size_t)d_inner * n, 0.0);
        Mat y(T, d_inner);
        for (int t = 0; t < T; ++t) {
            const double* Bt = dbc.row(t) + dt_rank;
            const double* Ct = dbc.row(t) + dt_rank + n;
            const double* dt = delta.row(t);
            const double* xt = xc.row(t);
            double* yt = y.row(t);
            for (int c = 0; c < d_inner; ++c) {
                const double* Ac = A_log.row(c);
                double* hc = h.data() + (size_t)c * n;
                double acc = 0.0;
                for (int j = 0; j < n; ++j) {
                    double abar = std::exp(dt[c] * (-std::exp(Ac[j])));
                    hc[j] = abar * hc[j] + (dt[c] * Bt[j]) * xt[c];
                    acc += hc[j] * Ct[j];
                }
                yt[c] = acc + D[c] * xt[c];
            }
        }
        for (size_t i = 0; i < y.v.size(); ++i) y.v[i] *= silu_(z.v[i]);
        return linear(y, out_proj, nullptr);
    }
};

struct Memory {                                    // TypedRecurrentMemory
    int S;                                         // 7 * channel_dim
    Mat fw, ww, rw, cw, ro; vector<double> fb, wb, rb, cb, rob, retention;
    void load(Reader& r, int d_model, int channel_dim) {
        S = 7 * channel_dim;
        fw = r.mat(S, d_model + S); fb = r.vec(S);
        ww = r.mat(S, d_model + S); wb = r.vec(S);
        rw = r.mat(S, d_model + S); rb = r.vec(S);
        cw = r.mat(S, d_model + S); cb = r.vec(S);
        ro = r.mat(d_model, S); rob = r.vec(d_model);
        const double lam[7] = {0.98, 0.97, 0.95, 0.99, 0.90, 0.999, 0.98};
        retention.resize(S);                       // fixed constant, not exported
        for (int k = 0; k < 7; ++k)
            for (int j = 0; j < channel_dim; ++j) retention[k * channel_dim + j] = lam[k];
    }
    Mat fwd(const Mat& units) const {
        int P = units.r, d = units.c;
        vector<double> c(S, 0.0), h(S, 0.0), zin(d + S), f(S), w(S), rr(S), dc(S);
        Mat out(P, d);
        for (int p = 0; p < P; ++p) {
            std::memcpy(zin.data(), units.row(p), (size_t)d * 8);
            std::memcpy(zin.data() + d, h.data(), (size_t)S * 8);
            auto gate = [&](const Mat& W, const vector<double>& B, vector<double>& dst, bool sig) {
                for (int o = 0; o < S; ++o) {
                    const double* wr = W.row(o);
                    double s = B[o];
                    for (int i = 0; i < d + S; ++i) s += zin[i] * wr[i];
                    dst[o] = sig ? sigmoid_(s) : std::tanh(s);
                }
            };
            gate(fw, fb, f, true); gate(ww, wb, w, true); gate(rw, rb, rr, true); gate(cw, cb, dc, false);
            for (int i = 0; i < S; ++i) {
                c[i] = retention[i] * f[i] * c[i] + w[i] * dc[i];   // authority a=1
                h[i] = rr[i] * std::tanh(c[i]);
            }
            double* op = out.row(p);
            for (int o = 0; o < d; ++o) {
                const double* wr = ro.row(o);
                double s = rob[o];
                for (int i = 0; i < S; ++i) s += h[i] * wr[i];
                op[o] = s;
            }
        }
        return out;
    }
};

struct HybridBlock {
    vector<double> n1, n2; SSM ssm; Attn local, sparse, retrieval; GatedMLP ffn; MLP2 router;
    int window, topk;
    void load(Reader& r, const Config& c) {
        window = c.unit_window; topk = std::max(4, c.p_max / 2);
        n1 = r.vec(c.d_model); n2 = r.vec(c.d_model);
        ssm.load(r, c.d_model, c.d_state);
        local.load(r, c.d_model, c.n_heads);
        sparse.load(r, c.d_model, c.n_heads);
        retrieval.load(r, c.d_model, c.n_heads);   // unused at inference (no retrieval)
        ffn.load(r);
        router.load(r);
    }
    void fwd(Mat& x, const vector<uint8_t>* kp) const {
        Mat h = x; rmsnorm(h, n1);
        Mat ys = ssm.fwd(h, kp);
        Mat yl = self_attention(local, h, window, kp, -1);
        Mat yg = self_attention(sparse, h, -1, kp, topk);
        // y_ret = zeros (retrieval-free path)
        int T = x.r, d = x.c;
        Mat route(T, 4 * d);
        for (int t = 0; t < T; ++t) {
            std::memcpy(route.row(t),         ys.row(t), (size_t)d * 8);
            std::memcpy(route.row(t) + d,     yl.row(t), (size_t)d * 8);
            std::memcpy(route.row(t) + 2 * d, yg.row(t), (size_t)d * 8);
            // route[3d:4d] stays zero
        }
        Mat alpha = router.fwd(route);             // [T,4]
        for (int t = 0; t < T; ++t) {              // oracle softmax: max-shift
            double* a = alpha.row(t);
            double m = std::max(std::max(a[0], a[1]), std::max(a[2], a[3]));
            double s = 0.0;
            for (int i = 0; i < 4; ++i) { a[i] = std::exp(a[i] - m); s += a[i]; }
            for (int i = 0; i < 4; ++i) a[i] /= s;
        }
        for (int t = 0; t < T; ++t) {
            const double* a = alpha.row(t);
            double* xt = x.row(t);
            for (int i = 0; i < d; ++i)
                xt[i] += a[0] * ys.row(t)[i] + a[1] * yl.row(t)[i] + a[2] * yg.row(t)[i];
        }
        Mat h2 = x; rmsnorm(h2, n2);
        Mat ff = ffn.fwd(h2);
        for (size_t i = 0; i < x.v.size(); ++i) x.v[i] += ff.v[i];
    }
};

struct DecoderLayer {
    vector<double> n1, n2, nr, n3; Attn self_a, cross_a, ret_a; GatedMLP ffn; int window;
    void load(Reader& r, const Config& c) {
        window = c.decoder_window;
        n1 = r.vec(c.d_model); self_a.load(r, c.d_model, c.n_heads);
        n2 = r.vec(c.d_model); cross_a.load(r, c.d_model, c.n_heads);
        nr = r.vec(c.d_model); ret_a.load(r, c.d_model, c.n_heads);  // unused
        n3 = r.vec(c.d_model); ffn.load(r);
    }
    void fwd(Mat& x, const Mat& units, const vector<uint8_t>& cross_mask) const {
        Mat h = x; rmsnorm(h, n1);
        Mat sa = self_attention(self_a, h, window, nullptr, -1);
        for (size_t i = 0; i < x.v.size(); ++i) x.v[i] += sa.v[i];
        Mat h2 = x; rmsnorm(h2, n2);
        Mat ca = cross_attention(cross_a, h2, units, cross_mask);
        for (size_t i = 0; i < x.v.size(); ++i) x.v[i] += ca.v[i];
        Mat h3 = x; rmsnorm(h3, n3);
        Mat ff = ffn.fwd(h3);
        for (size_t i = 0; i < x.v.size(); ++i) x.v[i] += ff.v[i];
    }
};

// -------------------------------------------------------------------- model
struct Model {
    Config c;
    // perception
    Mat byte_embed; Mat c3w, c7w, cdw; vector<double> c3b, c7b, cdb, pnorm;
    MLP2 boundary_head;
    // units
    Mat ue_w; vector<double> ue_b, ue_norm;
    Memory memory;
    vector<HybridBlock> blocks;
    // decoder
    Mat dec_embed; vector<DecoderLayer> dlayers; vector<double> dnorm;
    Mat head_w; vector<double> head_b;
    // tail (unused at inference but present in export order)
    Mat ret_byte_embed, ret_proj_w, bproj_w; vector<double> ret_proj_b, bproj_b;

    void load(const std::string& path) {
        Reader r(path);
        assert(r.get<uint32_t>() == 0x414E4643u);
        assert(r.get<uint32_t>() == 1u);
        int* cf = &c.byte_embed_dim;
        for (int i = 0; i < 15; ++i) cf[i] = r.get<int32_t>();
        c.tau = r.get<double>();
        c.trainable_segmentation = r.get<uint8_t>() != 0;
        uint32_t nt = r.get<uint32_t>(); (void)nt;
        // ---- exact parameters() order (see cpp/README.md) ----
        byte_embed = r.mat(256, c.byte_embed_dim);
        vector<int> s;
        c3w = Mat(); c3w.v = r.raw(s); c3w.r = s[0]; c3w.c = s[1] * s[2];  // [dl, emb*3]
        c3b = r.vec(c.d_local);
        c7w = Mat(); c7w.v = r.raw(s); c7w.r = s[0]; c7w.c = s[1] * s[2];
        c7b = r.vec(c.d_local);
        cdw = Mat(); cdw.v = r.raw(s); cdw.r = s[0]; cdw.c = s[1] * s[2];
        cdb = r.vec(c.d_local);
        pnorm = r.vec(c.d_local);
        boundary_head.load(r);
        ue_w = r.mat(c.d_model, c.d_local); ue_b = r.vec(c.d_model);
        ue_norm = r.vec(c.d_model);
        memory.load(r, c.d_model, c.channel_dim);
        blocks.resize(c.physical_blocks);
        for (auto& b : blocks) b.load(r, c);
        dec_embed = r.mat(256, c.d_model);
        dlayers.resize(c.decoder_layers);
        for (auto& l : dlayers) l.load(r, c);
        dnorm = r.vec(c.d_model);
        head_w = r.mat(256, c.d_model); head_b = r.vec(256);
        ret_byte_embed = r.mat(256, c.ret_byte_dim);
        ret_proj_w = r.mat(c.d_model, c.d_local + c.ret_byte_dim); ret_proj_b = r.vec(c.d_model);
        bproj_w = r.mat(c.d_model, 1); bproj_b = r.vec(c.d_model);
    }

    // conv over [T, Cin] features with weight rows [Cout, Cin*K] (K taps
    // ordered as exported: w[co][ci][i], left-padded causal, dilation dil).
    static Mat conv_causal(const Mat& x, const Mat& w, const vector<double>& b,
                           int K, int dil) {
        int T = x.r, cin = x.c, cout = w.r;
        Mat y(T, cout);
        for (int t = 0; t < T; ++t) {
            double* yt = y.row(t);
            for (int co = 0; co < cout; ++co) {
                const double* wr = w.row(co);
                double s = b[co];
                for (int i = 0; i < K; ++i) {
                    int src = t - (K - 1 - i) * dil;
                    if (src < 0) continue;
                    const double* xs = x.row(src);
                    const double* wi = wr + i;      // stride K over ci
                    for (int ci = 0; ci < cin; ++ci) s += wi[(size_t)ci * K] * xs[ci];
                }
                yt[co] = s;
            }
        }
        return y;
    }

    // perception: bytes -> (feats [T,d_local], boundary probs [T])
    void perception(const vector<int>& ids, Mat& feats, vector<double>& prob) const {
        int T = (int)ids.size();
        Mat e(T, c.byte_embed_dim);
        for (int t = 0; t < T; ++t)
            std::memcpy(e.row(t), byte_embed.row(ids[t]), (size_t)c.byte_embed_dim * 8);
        Mat x = conv_causal(e, c3w, c3b, 3, 1);
        for (double& t : x.v) t = gelu_(t);
        Mat x2 = conv_causal(x, c7w, c7b, 7, 1);
        for (size_t i = 0; i < x.v.size(); ++i) x.v[i] += gelu_(x2.v[i]);
        Mat x3 = conv_causal(x, cdw, cdb, 3, 4);
        for (size_t i = 0; i < x.v.size(); ++i) x.v[i] += gelu_(x3.v[i]);
        feats = x; rmsnorm(feats, pnorm);
        Mat bl = boundary_head.fwd(feats);          // [T,1]
        prob.resize(T);
        for (int t = 0; t < T; ++t) prob[t] = sigmoid_(bl.row(t)[0]);
    }

    // build unit stack from pooled means (+ mean boundary prob), padded to p_max
    Mat unit_stack(const vector<vector<double>>& pooled, const vector<double>& mprob,
                   vector<uint8_t>& unit_mask) const {
        Mat u(c.p_max, c.d_local);
        for (size_t p = 0; p < pooled.size(); ++p)
            std::memcpy(u.row((int)p), pooled[p].data(), (size_t)c.d_local * 8);
        Mat units = linear(u, ue_w, &ue_b);
        rmsnorm(units, ue_norm);
        if (c.trainable_segmentation)
            for (int p = 0; p < c.p_max; ++p) {
                double mp = p < (int)mprob.size() ? mprob[p] : 0.0;
                double* r_ = units.row(p);
                for (int i = 0; i < c.d_model; ++i) r_[i] += bproj_w.row(i)[0] * mp + bproj_b[i];
            }
        Mat mem = memory.fwd(units);
        for (size_t i = 0; i < units.v.size(); ++i) units.v[i] += mem.v[i];
        unit_mask.assign(c.p_max, 0);
        for (size_t p = 0; p < pooled.size(); ++p) unit_mask[p] = 1;
        for (int t = 0; t < c.logical_depth; ++t)
            blocks[t % blocks.size()].fwd(units, &unit_mask);
        return units;
    }

    Mat decode(const vector<int>& ids, const Mat& units, const vector<int>& seg) const {
        int T = (int)ids.size();
        Mat x(T, c.d_model);
        for (int t = 0; t < T; ++t)
            std::memcpy(x.row(t), dec_embed.row(ids[t]), (size_t)c.d_model * 8);
        vector<uint8_t> cross((size_t)T * c.p_max, 0);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < c.p_max && j < seg[t]; ++j) cross[(size_t)t * c.p_max + j] = 1;
        for (const auto& l : dlayers) l.fwd(x, units, cross);
        rmsnorm(x, dnorm);
        return linear(x, head_w, &head_b);          // [T,256]
    }
};

// -------------------------------------------------- dense forward (parity)
// Mirrors MicroCFNAModel.forward exactly, INCLUDING the trailing partial unit.
static vector<double> dense_last_logits(const Model& M, const vector<int>& ids) {
    Mat feats; vector<double> prob;
    M.perception(ids, feats, prob);
    int T = (int)ids.size();
    vector<int> seg(T, 0);
    int cur = 0, len = 1;
    for (int i = 1; i < T; ++i) {
        bool cut = (prob[i] > M.c.tau && len >= M.c.min_patch) || len >= M.c.max_patch;
        if (cut) { cur = std::min(cur + 1, M.c.p_max - 1); len = 1; } else ++len;
        seg[i] = cur;
    }
    int n_units = cur + 1;
    vector<vector<double>> pooled(n_units, vector<double>(M.c.d_local, 0.0));
    vector<double> mp(n_units, 0.0);
    vector<int> cnt(n_units, 0);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < M.c.d_local; ++i) pooled[seg[t]][i] += feats.row(t)[i];
        mp[seg[t]] += prob[t]; ++cnt[seg[t]];
    }
    for (int p = 0; p < n_units; ++p) {
        for (int i = 0; i < M.c.d_local; ++i) pooled[p][i] /= cnt[p];
        mp[p] /= cnt[p];
    }
    vector<uint8_t> umask;
    Mat units = M.unit_stack(pooled, mp, umask);
    Mat logits = M.decode(ids, units, seg);
    return vector<double>(logits.row(T - 1), logits.row(T - 1) + 256);
}

// ------------------------------------------- incremental generation (fast)
// Port of cfna/microtorch/incremental.py (unit cache + windowed recompute);
// greedy output proven byte-identical to the dense path by the Python twin's
// test suite and re-checked here in tests/test_cpp_engine.py.
struct Incremental {
    const Model& M;
    int per_span, dec_span;
    vector<int> ids, seg;
    int cur = 0, len = 1;
    vector<vector<double>> done_feats; vector<double> done_prob;
    vector<double> acc_f; double acc_p = 0; int acc_n = 0;
    Mat units; vector<uint8_t> umask; bool dirty = true;

    explicit Incremental(const Model& m) : M(m) {
        per_span = 17 + 8;
        dec_span = M.c.decoder_layers * M.c.decoder_window + 8;
    }
    void advance(const double* feat, double p) {
        bool cut = (p > M.c.tau && len >= M.c.min_patch) || len >= M.c.max_patch;
        if (cut) {
            if (cur < M.c.p_max - 1) {
                vector<double> mean(M.c.d_local);
                for (int i = 0; i < M.c.d_local; ++i) mean[i] = acc_f[i] / acc_n;
                done_feats.push_back(mean);
                done_prob.push_back(acc_p / acc_n);
                acc_f.assign(feat, feat + M.c.d_local); acc_p = p; acc_n = 1;
                ++cur; dirty = true;
            } else { for (int i = 0; i < M.c.d_local; ++i) acc_f[i] += feat[i]; acc_p += p; ++acc_n; }
            len = 1;
        } else { for (int i = 0; i < M.c.d_local; ++i) acc_f[i] += feat[i]; acc_p += p; ++acc_n; ++len; }
        seg.push_back(cur);
    }
    void prime(const vector<int>& in) {
        ids = in; seg.assign(1, 0); cur = 0; len = 1;
        done_feats.clear(); done_prob.clear();
        Mat feats; vector<double> prob;
        M.perception(ids, feats, prob);
        acc_f.assign(feats.row(0), feats.row(0) + M.c.d_local);
        acc_p = prob[0]; acc_n = 1;
        for (int i = 1; i < (int)ids.size(); ++i) advance(feats.row(i), prob[i]);
        rebuild();
    }
    void rebuild() { units = M.unit_stack(done_feats, done_prob, umask); dirty = false; }
    vector<double> last_logits() {
        if (dirty) rebuild();
        int span = std::min((int)ids.size(), dec_span);
        vector<int> wi(ids.end() - span, ids.end());
        vector<int> ws(seg.end() - span, seg.end());
        // clamp cross indices to completed units (seg already < p_max)
        Mat logits = M.decode(wi, units, ws);
        return vector<double>(logits.row(span - 1), logits.row(span - 1) + 256);
    }
    void append(int b) {
        ids.push_back(b);
        int span = std::min((int)ids.size(), per_span);
        vector<int> wi(ids.end() - span, ids.end());
        Mat feats; vector<double> prob;
        M.perception(wi, feats, prob);
        advance(feats.row(span - 1), prob[span - 1]);
    }
};

// ----------------------------------------------------------------- driver
int main(int argc, char** argv) {
    std::string model_path, prompt;
    int max_new = 64, max_ctx = 256;
    bool logits_mode = false, greedy = true, bench = false, hex_out = false;
    double temperature = 0.0;
    unsigned seed = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--prompt") prompt = argv[++i];
        else if (a == "--max-new") max_new = std::atoi(argv[++i]);
        else if (a == "--max-ctx") max_ctx = std::atoi(argv[++i]);
        else if (a == "--logits") logits_mode = true;
        else if (a == "--temp") { temperature = std::atof(argv[++i]); greedy = temperature <= 0.0; }
        else if (a == "--seed") seed = (unsigned)std::atoi(argv[++i]);
        else if (a == "--bench") bench = true;
        else if (a == "--hex") hex_out = true;   // print output as hex (test-safe)
        else model_path = a;
    }
    if (model_path.empty()) {
        std::fprintf(stderr,
            "usage: cfna_run model.bin --prompt TEXT [--max-new N] [--max-ctx N]\n"
            "       [--temp T] [--seed S] [--logits] [--bench]\n");
        return 2;
    }
    Model M;
    M.load(model_path);
    vector<int> ids;
    for (unsigned char ch : prompt) ids.push_back((int)ch);
    if (ids.empty()) ids.push_back(32);

    if (logits_mode) {                              // parity interface
        vector<int> win(ids.end() - std::min<size_t>(ids.size(), max_ctx), ids.end());
        vector<double> l = dense_last_logits(M, win);
        for (int i = 0; i < 256; ++i) std::printf("%.17g\n", l[i]);
        return 0;
    }

    std::mt19937 rng(seed);
    Incremental inc(M);
    // The dense reference keeps the FULL prompt in its output and slices only
    // the per-step context window; mirror that exactly.
    vector<int> out = ids;
    vector<int> win0(ids.end() - std::min<size_t>(ids.size(), max_ctx), ids.end());
    inc.prime(win0);
    auto t0 = clock();
    for (int stepn = 0; stepn < max_new; ++stepn) {
        vector<double> l = inc.last_logits();
        int idx;
        if (greedy) idx = (int)(std::max_element(l.begin(), l.end()) - l.begin());
        else {
            double m = *std::max_element(l.begin(), l.end()), s = 0.0;
            vector<double> p(256);
            for (int i = 0; i < 256; ++i) { p[i] = std::exp((l[i] - m) / temperature); s += p[i]; }
            std::uniform_real_distribution<double> u(0.0, s);
            double r = u(rng), acc2 = 0.0; idx = 255;
            for (int i = 0; i < 256; ++i) { acc2 += p[i]; if (r <= acc2) { idx = i; break; } }
        }
        out.push_back(idx);
        if ((int)inc.ids.size() >= max_ctx) {
            vector<int> win(out.end() - std::min<size_t>(out.size(), max_ctx), out.end());
            inc.prime(win);                          // dense path slides; re-prime
        } else inc.append(idx);
    }
    double secs = double(clock() - t0) / CLOCKS_PER_SEC;
    if (hex_out) {
        for (int b : out) std::printf("%02x", b);
        std::printf("\n");
    } else {
        std::string text(out.begin(), out.end());
        std::printf("%s\n", text.c_str());
    }
    if (bench) std::fprintf(stderr, "gen: %d bytes in %.3fs (%.1f bytes/s)\n",
                            max_new, secs, max_new / (secs > 0 ? secs : 1e-9));
    return 0;
}
