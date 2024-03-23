/*
    Free-space diffraction BSDF
    Copyright (c) 2023-2024 Shlomi Steinberg.
*/

#pragma once

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/scene.h>

#include <complex>


MTS_NAMESPACE_BEGIN


struct tri_t { Point2 a,b,c; };

// From boost
template<typename T>
inline T sinc(const T x) {
    T const taylor_0_bound = std::numeric_limits<T>::epsilon();
    T const taylor_2_bound = static_cast<T>(0.00034526698300124390839884978618400831996329879769945L);
    T const taylor_n_bound = static_cast<T>(0.018581361171917516667460937040007436176452688944747L);

    if (abs(x) >= taylor_n_bound)
        return sin(x)/x;
    // approximation by taylor series in x at 0 up to order 0
    T    result = static_cast<T>(1);
    if (abs(x) >= taylor_0_bound) {
        T    x2 = x*x;
        // approximation by taylor series in x at 0 up to order 2
        result -= x2/static_cast<T>(6);
        if    (abs(x) >= taylor_2_bound)
            result += (x2*x2)/static_cast<T>(120);
    }

    return result;
}

template <typename T>
inline auto sqr(const T t) noexcept { return t*t; }
template <typename T>
inline auto clamp(const T t, const T min, const T max) noexcept { return std::min(std::max(min,t),max); }
template <typename T>
inline auto clamp01(const T t) noexcept { return clamp(t,T(0),T(1)); }


inline int intersectSphereLine(const float r, const Point3& a, const Point3& b, Point3 &p1, Point3 &p2, float &t1, float &t2) {
    const auto &v = b-a;
    const auto A = v.length2();
    const auto B = 2.f * dot(Vector3{ a },v);
    const auto C = a.length2() - sqr(r);

    const float delta2 = sqr(B)-4.f*A*C;
    if (delta2<=0) { t1=0;t2=1; return 0; }

    const auto delta = sqrtf(delta2);
    t1 = (-B+delta) / (2.f*A);
    t2 = (-B-delta) / (2.f*A);
    if (t2<t1) std::swap(t1,t2);
    p1 = a+t1*v;
    p2 = a+t2*v;

    const bool p1v = t1>0 && t1<1;
    const bool p2v = t2>0 && t2<1;
    if (!p1v && !p2v) return 0;
    if (p1v && p2v) return 2;
    if (!p1v) p1=p2;
    return 1;
}
inline int intersectSphereLine(const float r, const Point3& a, const Point3& b, Point3 &p1, Point3 &p2) {
    Float t1,t2;
    return intersectSphereLine(r, a,b, p1,p2, t1,t2);
}
inline int intersectSphereLine(const float r, const Point3& a, const Point3& b, Float &t1, Float &t2) {
    Point3 p1,p2;
    return intersectSphereLine(r, a,b, p1,p2, t1,t2);
}
inline bool intersectSphereLine(const float r, const Point3& a, const Point3& b) {
    Point3 p1,p2;
    return intersectSphereLine(r, a,b, p1,p2)>0;
}

inline float areaTri(const Point2& a, const Point2& b) { return .5f*(a.x*b.y-b.x*a.y); }
inline int intersectCircleLine(const float r, const Point2& a, const Point2& b, Point2 &p1, Point2 &p2, float &t1, float &t2) {
    const auto d = b-a;
    const auto adotd = dot((Vector2)a,d);
    const float delta2 = 4*sqr(adotd)-4*d.length2()*(a.length2()-r*r);
    if (delta2<=0) { t1=0;t2=1; return 0; }

    const auto delta = sqrtf(delta2);
    t1 = (-adotd+.5f*delta) / d.length2();
    t2 = (-adotd-.5f*delta) / d.length2();
    if (t2<t1) std::swap(t1,t2);
    p1 = a+t1*d;
    p2 = a+t2*d;

    const bool p1v = t1>0 && t1<1;
    const bool p2v = t2>0 && t2<1;
    if (!p1v && !p2v) return 0;
    if (p1v && p2v) return 2;
    if (!p1v) p1=p2;
    return 1;
}
inline int intersectCircleLine(const float r, const Point2& a, const Point2& b, Point2 &p1, Point2 &p2) {
    Float t1,t2;
    return intersectCircleLine(r, a,b, p1,p2, t1,t2);
}
inline int intersectCircleLine(const float r, const Point2& a, const Point2& b, Float &t1, Float &t2) {
    Point2 p1,p2;
    return intersectCircleLine(r, a,b, p1,p2, t1,t2);
}
inline float areaCircSector(const float r, float theta) {
    theta = fabsf(theta);
    if (theta>M_PI) theta=2*M_PI-theta;
    return .5f*r*r*theta;
}
inline float areaCircSectorLine(const float r, const Point2& a, const Point2& b) {
    Point2 p1={0,0}, p2={0,0};
    const auto ps = intersectCircleLine(r,a,b,p1,p2);

    const auto thetaa = atan2f(a.y,a.x);
    const auto thetab = atan2f(b.y,b.x);
    const auto thetap1 = ps>0 ? atan2f(p1.y,p1.x) : .0f;
    const auto thetap2 = ps>1 ? atan2f(p2.y,p2.x) : .0f;
    auto wnd = thetaa-thetab;
    if (wnd<-M_PI) wnd+=2*M_PI;
    if (wnd>M_PI)  wnd-=2*M_PI;
    const auto sgn = wnd<.0f?-1.f:1.f;

    if (ps==2) {
        return sgn * (fabsf(areaTri(p1,p2)) + areaCircSector(r,thetaa-thetap1) + areaCircSector(r,thetap2-thetab));
    }
    else if (ps==1) {
        if (a.length2()<r*r) return sgn * (fabsf(areaTri(a,p1)) + areaCircSector(r,thetab-thetap1));
        else                 return sgn * (fabsf(areaTri(p1,b)) + areaCircSector(r,thetaa-thetap1));
    }
    const auto thetaab = thetaa-thetab;
    if (a.length2()<r*r) return sgn*fabsf(areaTri(a,b));
    else                 return sgn*areaCircSector(r,thetaab);
}
inline float areaCircleTri(const float r, const Point2& a, const Point2& b, const Point2& c) {
    return fabsf(areaCircSectorLine(r,a,b) +
                    areaCircSectorLine(r,b,c) +
                    areaCircSectorLine(r,c,a));
}

inline bool isPointInTri(const Point2& p, const tri_t &tri) {
    const float d1 = areaTri(Point2{ p-tri.b }, Point2{ tri.a-tri.b });
    const float d2 = areaTri(Point2{ p-tri.c }, Point2{ tri.b-tri.c });
    const float d3 = areaTri(Point2{ p-tri.a }, Point2{ tri.c-tri.a });

    const auto has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    const auto has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(has_neg && has_pos);
}


struct fsdPrecomputedTables {
    inline auto importanceSampleCDF1(const Point3 &rand3) const {
        return importanceSampleCDF(rand3, iCDFtheta1, iCDF1);
    }
    inline auto importanceSampleCDF2(const Point3 &rand3) const {
        return importanceSampleCDF(rand3, iCDFtheta2, iCDF2);
    }

private:
    static constexpr std::size_t Nsamples = 1024, Msamples = 1024;
    std::array<Float, Nsamples> iCDFtheta1, iCDFtheta2;
    std::array<std::array<Float, Msamples>, Msamples> iCDF1, iCDF2;

    template <std::size_t S>
    static inline auto lerp(Float x, const std::array<Float, S> &iCDFtheta) {
        x *= S;
        const auto l = std::min((std::size_t)x, S-1);
        const auto h = std::min(l+1, S-1);
        const auto f = std::max<Float>(1,x-(Float)l);
        return f*iCDFtheta[h] + (1-f)*iCDFtheta[l];
    }
    template <std::size_t S>
    static inline auto lerp(Float theta, Float rx, const std::array<std::array<Float, S>, S> &iCDF) {
        const auto x = theta*2/M_PI * S;
        const auto l = std::min((std::size_t)x, S-1);
        const auto h = std::min(l+1, S-1);
        const auto f = std::max<Float>(1,x-(Float)l);
        return f*lerp(rx,iCDF[h]) + (1-f)*lerp(rx,iCDF[l]);
    }
    
    static inline Vector2 importanceSampleCDF(const Point3 &rand3, 
                             const std::array<Float, Nsamples> &iCDFtheta,
                             const std::array<std::array<Float, Msamples>, Msamples> &iCDF) {
        const auto theta = lerp(rand3.x, iCDFtheta);
        const auto r = std::max((Float)0, lerp(theta, rand3.y, iCDF));
        const auto q = std::min(3,(int)(rand3.z*4));

        auto xi = r * Vector2{ cosf(theta), sinf(theta) };
        xi.x *= Float(((q+1)/2)%2==0?1:-1);
        xi.y *= Float((q/2)%2==0?1:-1);
        return xi;
    }

    static inline auto loadiCDFtheta(const boost::filesystem::path &path) {
        std::array<Float, Nsamples> data;

        std::ifstream f(path, std::ios::in);
        if (f.bad() || f.fail()) {
            SLog(EError, "Could not open \"%s\"", path.string().c_str());
            return data;
        }

        std::string str;
        for (auto &l : data) {
            if (!std::getline(f, str)) {
                assert(false);
                break;
            }
            std::stringstream conv(str);
            conv >> l;
        }

        return data;
    }
    static inline auto loadiCDF(const boost::filesystem::path &path) {
        std::array<std::array<Float, Msamples>, Msamples> data;

        std::ifstream f(path, std::ios::in);
        if (f.bad() || f.fail()) {
            SLog(EError, "Could not open \"%s\"", path.string().c_str());
            return data;
        }

        std::string line,str;
        for (auto &row : data) {
            if (!std::getline(f, line)) {
                assert(false);
                break;
            }
            std::stringstream ss(line);
            for (auto &cell : row) {
                if (!std::getline(ss, str, ',')) {
                    assert(false);
                    break;
                }
                std::stringstream conv(str);
                conv >> cell;
            }
        }

        return data;
    }
    void loadiCDFs() {
        ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();
        iCDFtheta1 = loadiCDFtheta(fResolver->resolve("data/fsd/iCDFa1theta.csv"));
        iCDFtheta2 = loadiCDFtheta(fResolver->resolve("data/fsd/iCDFa2theta.csv"));
        iCDF1 = loadiCDF(fResolver->resolve("data/fsd/iCDFa1.csv"));
        iCDF2 = loadiCDF(fResolver->resolve("data/fsd/iCDFa2.csv"));
    }

public:
    fsdPrecomputedTables() { loadiCDFs(); }
};


MTS_NAMESPACE_END
