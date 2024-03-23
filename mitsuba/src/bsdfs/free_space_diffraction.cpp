/*
    Free-space diffraction BSDF
    Copyright (c) 2023-2024 Shlomi Steinberg.
*/

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/matrix.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>

#include "fsdUtils.h"

#include <future>
#include <limits>
#include <complex>
#include <fstream>
#include <random>
#include <map>


MTS_NAMESPACE_BEGIN


static StatsCounter fsdStatFSD("FSD BRDF", "FSD performed out of BSDF constructions", EPercentage);


class FreeSpaceDiffractionBSDF : public BSDF {
public:
    static constexpr auto fsdLobe = 
        EGlossyTransmission
    ;
    static constexpr auto fsdMeasure = 
        ESolidAngle
    ;

    FreeSpaceDiffractionBSDF(const Properties &props)
        : BSDF(props) {
        m_enabled = props.getBoolean("enabled", true);
        m_SIR = props.getBoolean("SIR", true);
        m_sigma = props.getFloat("sigma", 25.f);
        // For debug
        m_scale = props.getFloat("scale", 1.f);
    }

    FreeSpaceDiffractionBSDF(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_bsdf = static_cast<BSDF *>(manager->getInstance(stream));
        m_enabled = stream->readBool();
        m_SIR = stream->readBool();
        m_sigma = stream->readFloat();
        m_scale = stream->readFloat();
        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);
        manager->serialize(stream, m_bsdf.get());
        stream->writeBool(m_enabled);
        stream->writeBool(m_SIR);
        stream->writeFloat(m_sigma);
        stream->writeFloat(m_scale);
    }

    void configure() {
        if (!m_bsdf)
            Log(EError, "A child BSDF instance is required");

        m_components.clear();
        for (int i=0; i<m_bsdf->getComponentCount(); ++i) {
            // Simplify implementation: don't allow transmission on nested
            if (!!(m_bsdf->getType(i) & ETransmission))
                Log(EError, "Transmissive nested BSDF unsupported");
            m_components.push_back(m_bsdf->getType(i));
        }

        // Free-space diffractions component
        m_components.push_back(ESpatiallyVarying | EAnisotropic | fsdLobe | EBackSide | EFrontSide | EUsesSampler | ENonSymmetric);

        if (m_scale>1)
            Log(EWarn, "Free-space diffraction BSDF does not conserve energy (scale>1)");
        
        m_usesRayDifferentials = true;
        
        BSDF::configure();
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(BSDF))) {
            if (m_bsdf != NULL)
                Log(EError, "Only a single nested BSDF can be added!");
            m_bsdf = static_cast<BSDF *>(child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    inline auto fsdComponent() const { return static_cast<int>(m_components.size()-1); }


    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        const bool evalNested = 
            bRec.component != fsdComponent() &&
            Frame::cosTheta(bRec.wi)*Frame::cosTheta(bRec.wo)>0;
        const bool evalFsd = 
            (bRec.component == -1 || bRec.component == fsdComponent()) && 
            measure==fsdMeasure && 
            Frame::cosTheta(bRec.wo)*Frame::cosTheta(bRec.wi)<0 && 
            !!(bRec.typeMask & fsdLobe);

        if (!m_enabled || evalNested)
            return m_bsdf->eval(bRec, measure);

        if (evalFsd) {
            if (!FSDpossible(bRec))
                return Spectrum{ .0f };

            // Evaluate possible free-space diffraction
            const auto wi = FSDwi(bRec), wo = FSDwo(bRec);
            int spec_idx;

            const auto& fsd = constructFSDBSDF(BSDF::getFrame(bRec.its), bRec.its.p, wi, Scene::globalScene, spec_idx);
            const auto ret = fsd.P()>0 ? fsd.bsdf(wo) : .0f;
            const auto intensity = std::max(.0f,ret);

            auto spectrum = Spectrum(.0f);
            spectrum[spec_idx] = intensity * m_scale;
            assert(!std::isnan(ret));
            return spectrum;
        }

        return Spectrum{ .0f };
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        const bool evalNested = 
            bRec.component != fsdComponent() &&
            Frame::cosTheta(bRec.wi)*Frame::cosTheta(bRec.wo)>0;
        const bool evalFsd = 
            (bRec.component == -1 || bRec.component == fsdComponent()) && 
            measure==fsdMeasure && 
            Frame::cosTheta(bRec.wo)*Frame::cosTheta(bRec.wi)<0 && 
            !!(bRec.typeMask & fsdLobe);

        if (!m_enabled || evalNested)
            return m_bsdf->pdf(bRec, measure);

        if (evalFsd) {
            if (!FSDpossible(bRec))
                return .0f;

            const auto wi = FSDwi(bRec), wo = FSDwo(bRec);
            int spec_idx;

            const auto& bsdf = constructFSDBSDF(BSDF::getFrame(bRec.its), bRec.its.p, wi, Scene::globalScene, spec_idx);

            return bsdf.P()>0 ? bsdf.pdf(wi,wo) : .0f;
        }

        return .0f;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
        if (!m_enabled)
            return m_bsdf->sample(bRec, pdf, sample);

        const bool fsdPossible = FSDpossible(bRec);
        auto fsdP = .0f; 
        bool sampleFSD = false;
        if (fsdPossible) {
            fsdP = .980f;
            sampleFSD = bRec.sampler->next1D()<fsdP;

            if (sampleFSD) {            
                int spec_idx;
                const auto &bsdf = constructFSDBSDF(BSDF::getFrame(bRec.its), bRec.its.p, bRec.its.toWorld(bRec.wi), Scene::globalScene, spec_idx);

                sampleFSD = bsdf.P()>0;

                fsdStatFSD.incrementBase(1);
                if (sampleFSD) {
                    ++fsdStatFSD;

                    Vector3 wo;
                    const auto ret = bsdf.importanceSample(wo, pdf, m_SIR, bRec.sampler);
                    if (pdf==0) return Spectrum(.0f);
                    const auto intensity = std::max(.0f,ret);

                    bRec.sampledNewP = bRec.its.p + bsdf.findExitPoint(bRec.sampler);
                    bRec.wo = bRec.its.toLocal(wo);
                    bRec.eta = 1.0f;
                    bRec.sampledComponent = fsdComponent();
                    bRec.sampledType = fsdLobe;

                    pdf *= fsdP;

                    assert(!std::isnan(ret));
                    auto spectrum = Spectrum(.0f);
                    spectrum[spec_idx] = intensity / fsdP * m_scale;
                    return spectrum;
                }
                else {
                    // FSD not possible
                    fsdP = 0.f;
                }
            }
        }

        if (!sampleFSD) {
            const auto ret = m_bsdf->sample(bRec, pdf, sample) / (1-fsdP);
            pdf *= 1-fsdP;
            bRec.sampledNewP = bRec.its.p;
            return ret;
        }

        pdf = .0f;
        return Spectrum{ .0f };
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        Float pdf;
        return this->sample(bRec, pdf, sample);
    }

    Point3 evalInteractionPoint(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        if (!m_enabled)
            return { 0,0,0 };

        const bool evalFsd = 
            (bRec.component == -1 || bRec.component == fsdComponent()) && 
            measure==fsdMeasure && 
            Frame::cosTheta(bRec.wi)*Frame::cosTheta(bRec.wo)<0 && 
            !!(bRec.typeMask & fsdLobe);

        if (evalFsd && FSDpossible(bRec)) {
            int spec_idx;
            const auto bsdf = constructFSDBSDF(BSDF::getFrame(bRec.its), bRec.its.p, bRec.its.toWorld(bRec.wi), Scene::globalScene, spec_idx);
            if (bsdf.P()==0)
                return bRec.its.p;

            return bRec.its.p + bsdf.findExitPoint(bRec.sampler);
        }

        return bRec.its.p;
    }
    bool updatesInteractionPoint(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        const bool evalFsd = 
            (bRec.component == -1 || bRec.component == fsdComponent()) && 
            measure==fsdMeasure && 
            !!(bRec.typeMask & fsdLobe);

        return m_enabled && evalFsd && FSDpossible(bRec);
    };
    bool updatesInteractionPoint() const { return m_enabled; }

    Vector3 FSDwi(const BSDFSamplingRecord &bRec) const {
        return bRec.its.toWorld(/*bRec.mode==ERadiance ? bRec.wo :*/ bRec.wi);
    }
    Vector3 FSDwo(const BSDFSamplingRecord &bRec) const {
        return bRec.its.toWorld(/*bRec.mode==ERadiance ? bRec.wi :*/ bRec.wo);
    }

    bool FSDpossible(const BSDFSamplingRecord &bRec) const {
        // A little hacky
        const float wavelength = Spectrum::m_wavelengths[0];
        return dot(bRec.wi,-bRec.wo)>0.35f && Vector3{ bRec.prevP - bRec.its.p }.length()>20*wavelength;
    }

    using c_t = std::complex<Float>;


private:
    // Free-space diffractions brains
    // In/out wi and wo are in world frames
    struct fsdBsdf {
    private:
        struct edge_t {
            Vector2 e;  // Edge vector
            Point2 v;   // Mid point
            c_t a,b;  // Beam amplitude at vertices
            Float Phat; // Edge-diffracted power
            Float PhatAccum;
        };

        // Alpha functions and Psi
        static inline Float alpha1(Float x,Float y) { return x==0 ? 0 : y/(x*x+y*y) * INV_PI * (cos(x/2)-sinc(x/2))/(2*x); }
        static inline Float alpha2(Float x,Float y) { return x==0 ? 0 : y/(x*x+y*y) * INV_PI * sinc(x/2)/2; }
        static inline Float chi(Float r2) { return sqrtf(1-exp(-.5f*r2/3)); }
        static inline c_t Psihat(c_t a, c_t b, Vector2 e, Point2 v, Float k, Vector2 xi) {
            const Float ee = e.length();
            const Float vxi = dot((Vector2)v,xi);
            const auto m = Vector2{ e.y,-e.x };

            xi = k * Vector2{ dot(e,xi), dot(m,xi) };
            const float chi0 = chi(dot(xi,xi));
            const c_t a1 = (a-b) *     alpha1(xi.x,xi.y);
            const c_t a2 = (a+b)/2.f * alpha2(xi.x,xi.y);

            return k * ee*ee * chi0 * std::polar(1.f, -k*vxi) * (a1 + c_t{0,1}*a2);
        }
        static inline Float Psihat2(c_t a, c_t b, Vector2 e, Float k, Vector2 xi) {
            const Float ee = e.length();
            const auto m = Vector2{ e.y,-e.x };

            xi = k * Vector2{ dot(e,xi), dot(m,xi) };
            const float chi0 = chi(dot(xi,xi));
            const c_t a1 = (a-b) *     alpha1(xi.x,xi.y);
            const c_t a2 = (a+b)/2.f * alpha2(xi.x,xi.y);

            return sqr(k * ee*ee * chi0) * std::norm(a1 + c_t{0,1}*a2);
        }

        // Powers
        static inline Float Pt(Point2 u1, Point2 u2, Point2 u3, Float ph1, Float ph2, Float ph3) {
            return (
                fabs(-u1.y*u2.x + u1.x*u2.y + u1.y*u3.x - u2.y*u3.x - u1.x*u3.y + u2.x*u3.y) *
                (sqr(ph3) + sqr(ph2) + sqr(ph1) + ph2*ph3 + ph1*ph2 + ph1*ph3)
                ) / Float(12);
        }
        static inline Float Pj(c_t a, c_t b, Vector2 e) { 
            return e.lengthSquared() * (std::norm(a-b) * 0.004973515f + std::norm(a+b)/4.f * 14.2569085397f);
        }
        static inline Float Pjhat(c_t a, c_t b, Vector2 e) { 
            return e.lengthSquared() * (std::norm(a-b) * 0.0045255085f + std::norm(a+b)/4.f * 0.114875434f);
        }

        static inline Float Psi0t(Point2 u1, Point2 u2, Point2 u3, Float ph1, Float ph2, Float ph3) {
            return ((ph1 + ph2 + ph3)*fabs(-(u1.y*u2.x) + u1.x*u2.y + u1.y*u3.x - u2.y*u3.x - u1.x*u3.y + u2.x*u3.y))/Float(6);
        }
        // 0-th order peak covariance
        static inline Vector3 Sigmat(Point2 u1, Point2 u2, Point2 u3, Float ph1, Float ph2, Float ph3) {
            const auto u1x=u1.x,u1y=u1.y,u2x=u2.x,u2y=u2.y,u3x=u3.x,u3y=u3.y;
            const auto a = (
                ((3*ph1 + ph2 + ph3)*sqr(u1x) + (ph1 + 3*ph2 + ph3)*sqr(u2x) + (ph1 + 2*(ph2 + ph3))*u2x*u3x + (ph1 + ph2 + 3*ph3)*sqr(u3x) + u1x*((2*(ph1 + ph2) + ph3)*u2x + (2*ph1 + ph2 + 2*ph3)*u3x))*fabs(-(u1y*u2x) + u1x*u2y + u1y*u3x - u2y*u3x - u1x*u3y + u2x*u3y)
            ) / Float(60);
            const auto b = (
                (u1x*(2*(3*ph1 + ph2 + ph3)*u1y + (2*(ph1 + ph2) + ph3)*u2y + (2*ph1 + ph2 + 2*ph3)*u3y) + u3x*((2*ph1 + ph2 + 2*ph3)*u1y + (ph1 + 2*(ph2 + ph3))*u2y + 2*(ph1 + ph2 + 3*ph3)*u3y) + u2x*((2*(ph1 + ph2) + ph3)*u1y + 2*(ph1 + 3*ph2 + ph3)*u2y + (ph1 + 2*(ph2 + ph3))*u3y))*fabs(-(u1y*u2x) + u1x*u2y + u1y*u3x - u2y*u3x - u1x*u3y + u2x*u3y)
            ) / Float(120);
            const auto c = (
                ((3*ph1 + ph2 + ph3)*sqr(u1y) + (ph1 + 3*ph2 + ph3)*sqr(u2y) + (ph1 + 2*(ph2 + ph3))*u2y*u3y + (ph1 + ph2 + 3*ph3)*sqr(u3y) + u1y*((2*(ph1 + ph2) + ph3)*u2y + (2*ph1 + ph2 + 2*ph3)*u3y))*fabs(-(u1y*u2x) + u1x*u2y + u1y*u3x - u2y*u3x - u1x*u3y + u2x*u3y)
            ) / Float(60);

            return { Float(a),Float(b),Float(c) };
        }

        inline Float eval(const Vector2& xi) const {
            if (P()<=0)
                return 0;

            const auto cosine = Float(1) / sqrtf(1+dot(xi,xi));
            c_t bsdf = {};
            for (const auto &e : edges) 
                bsdf += Psihat(e.a, e.b, e.e, e.v, k, xi);
            return 1.f/(cosine*P()) * std::norm(bsdf);
        }

        inline Float evalPdf(const Vector2& xi) const {
            if (sumPhat_j==0)
                return 0;

            Float pdf = 0;
            for (const auto &e : edges) 
                pdf += Psihat2(e.a, e.b, e.e, k, xi);
            return pdf/sumPhat_j;
        }


    public:
        const Float get_search_radius() const { return 3*beam_sigma; }

        fsdBsdf(const Scene *scene, const Point &p, const Vector3 &wi, const Frame &frame, const Float k, const Float beam_sigma) {
            // Incident beam profile
            const auto phi = [sigma=beam_sigma](const Point2 p,const Float z) -> Float 
                { return expf(-.25f*(p.length2()+sqr(z))/sqr(sigma))/(sqrtf(2*M_PI)*sigma); };

            this->k = k;
            this->beam_sigma = beam_sigma;
            Ppl_A = 0;
            sumPhat_j = 0;

            // Tangent to wi
            Vector tangent = frame.t - dot(wi,frame.t)*wi;
            {
                const auto tangentl = tangent.length();
                tangent = tangentl<1e-7f ? normalize(cross(Vector3{ wi.y,-wi.z,wi.x },wi)) :
                    tangent / tangentl;
            }
            const Vector bitangent = cross(wi,tangent);
            // Construct local tangent frame
            this->frame = Frame(tangent,bitangent,wi);


            // Get triangles
            const auto search_radius = get_search_radius();
            const auto Ts = scene->getKDTree()->getPrimitivesInSphere(p, search_radius);


            {
                // Check for early exit
                float totalArea = .0f;
                for (const auto& mt : Ts) {
                    const auto &t = mt.second;
                    if (dot(wi,t.n)<=0) continue;
    
                    // Project vertices
                    const auto u1 = Point2{ dot(tangent,  t.u1-p),
                                            dot(bitangent,t.u1-p) };
                    const auto u2 = Point2{ dot(tangent,  t.u2-p),
                                            dot(bitangent,t.u2-p) };
                    const auto u3 = Point2{ dot(tangent,  t.u3-p),
                                            dot(bitangent,t.u3-p) };
    
                    totalArea += areaCircleTri(search_radius, u1,u2,u3);
                }
                totalArea /= M_PI*sqr(search_radius);
                if (totalArea>1-1e-6f||totalArea<1e-6f) {
                    Ppl_A = 0;
                    return;
                }
            }


            // Find edges utility variables and functions
            Vector3 Sigmat_0 = { 0,0,0 };
            Float psi0 = 0;
            Float eavg = 0;

            const auto winding = [](const auto e, const auto v, const auto c) -> Float {
                const auto m = Vector2{ e.y,-e.x };
                return dot(m,v-c)>0?1:-1;
            };
            const auto addedge = [this,&eavg,&winding,&k](auto u1, auto u2, auto z1, auto z2, auto a, auto b, auto tric) {
                const auto v = (u1+u2)/Float(2);
                auto e = u2-u1;
                if (winding(e,v,tric)==-1) {
                    e=-e;
                    std::swap(a,b);
                    std::swap(z1,z2);
                }

                const auto ca = std::polar(a,k*z1);
                const auto cb = std::polar(b,k*z2);

                const auto P = Pjhat(ca,cb,e);
                if (P==0) return;
                this->sumPhat_j += P;
                this->edges.emplace_back(edge_t{ e,v,ca,cb,P,this->sumPhat_j });
                eavg += P*e.length();
            };

            const auto maxlength2 = sqr(beam_sigma);
            std::function<void(bool,bool,bool, const Point2&,const Point2&,const Point2&, Float, Float, Float, int)> addtri;
            addtri = [&,beam_sigma,search_radius,maxlength2,max_depth=5](bool edge12, bool edge13, bool edge23, const Point2& u1, const Point2& u2, const Point2& u3, Float z1, Float z2, Float z3, int recr_depth) -> void {
                if (areaCircleTri(search_radius, u1,u2,u3)<1e-10f)
                    return;

                const auto c = (u1+u2+u3)/3.f;
                const auto z0 = (z1+z2+z3)/3.f;

                const bool subdivide12 = (u2-u1).length2()>maxlength2;
                const bool subdivide13 = (u3-u1).length2()>maxlength2;
                const bool subdivide23 = (u3-u2).length2()>maxlength2;
                const auto sss = ((int)subdivide12)+((int)subdivide13)+((int)subdivide23)==1;

                // Subdivide triangles if needed. If only one edge is too long, subdivide along that edge only, otherwise split in the centre as well.
                if (recr_depth<max_depth && (subdivide12 || subdivide13 || subdivide23)) {
                    if (subdivide12) {
                        addtri(edge12,sss&&edge13,false, u1,(u1+u2)/2,sss?u3:c, z1,(z1+z2)/2,sss?z3:z0, recr_depth+1);
                        addtri(edge12,false,sss&&edge23, (u1+u2)/2,u2,sss?u3:c, (z1+z2)/2,z2,sss?z3:z0, recr_depth+1);
                    }
                    else if (!sss)
                        addtri(edge12,false,false, u1,u2,c, z1,z2,z0, recr_depth+1);
                    if (subdivide13) {
                        addtri(sss&&edge12,edge13,false, u1,sss?u2:c,(u1+u3)/2, z1,sss?z2:z0,(z1+z3)/2, recr_depth+1);
                        addtri(false,edge13,sss&&edge23, (u1+u3)/2,sss?u2:c,u3, (z1+z3)/2,sss?z2:z0,z3, recr_depth+1);
                    }
                    else if (!sss)
                        addtri(false,edge13,false, u1,c,u3, z1,z0,z3, recr_depth+1);
                    if (subdivide23) {
                        addtri(false,sss&&edge13,edge23, sss?u1:c,(u2+u3)/2,u3, sss?z1:z0,(z2+z3)/2,z3, recr_depth+1);
                        addtri(sss&&edge12,false,edge23, sss?u1:c,u2,(u2+u3)/2, sss?z1:z0,z2,(z2+z3)/2, recr_depth+1);
                    }
                    else if (!sss)
                        addtri(false,false,edge23, c,u2,u3, z0,z2,z3, recr_depth+1);
                    return;
                }

                const auto ph1 = phi(u1,z1);
                const auto ph2 = phi(u2,z2);
                const auto ph3 = phi(u3,z3);

                if (edge12) addedge(u2,u1,z2,z1,ph2,ph1,c);
                if (edge13) addedge(u1,u3,z1,z3,ph1,ph3,c);
                if (edge23) addedge(u3,u2,z3,z2,ph3,ph2,c);

                // Bookkeeping
                Ppl_A += Pt(u1,u2,u3,ph1,ph2,ph3);
                Sigmat_0 += Sigmat(u1,u2,u3,ph1,ph2,ph3);
                psi0 += Psi0t(u1,u2,u3,ph1,ph2,ph3);
            };

            // Find edges
            edges.reserve(100*Ts.size());   // Extra large in case of tessellation
            for (const auto& mt : Ts) {
                const auto &t = mt.second;
                if (dot(wi,t.n)<=0) continue;

                // Project vertices
                const auto u1 = Point2{ dot(tangent,  t.u1-p),
                                        dot(bitangent,t.u1-p) };
                const auto u2 = Point2{ dot(tangent,  t.u2-p),
                                        dot(bitangent,t.u2-p) };
                const auto u3 = Point2{ dot(tangent,  t.u3-p),
                                        dot(bitangent,t.u3-p) };
                const auto z1 = dot(-wi,t.u1-p);
                const auto z2 = dot(-wi,t.u2-p);
                const auto z3 = dot(-wi,t.u3-p);

                addtri(dot(wi,t.nn3)<=0,dot(wi,t.nn2)<=0,dot(wi,t.nn1)<=0,u1,u2,u3,z1,z2,z3,0);
            }

            // TODO: Merge edges to avoid very small edges.

            edges.shrink_to_fit();

            if (!edges.size() || P()<1e-2f) {
                Ppl_A = 0;
                return;
            }

            // Power in 0-th order lobe
            eavg /= sumPhat_j;
            const Float sigma_xi = sqrt(3)/(k*eavg);
            Sigmat_0 *= 6*k*k/psi0;
            const auto Sigma0 = Sigmat_0 + Vector3{ 1/sqr(sigma_xi),0,1/sqr(sigma_xi) };
            const auto detSigma0 = std::max<Float>(0, Sigma0.x*Sigma0.z-sqr(Sigma0.y));
            Ppl_0 = k*k/(18*M_PI) / sqrtf(detSigma0) * sqr(psi0);
        }

        Float importanceSample(Vector3 &wo, Float &pdf, const bool SIR, Sampler *sampler) const {
            Float bsdf;
            Vector2 xi;

            if (edges.size()==1 || !SIR) {
                xi = sampleEdge(edges[0],pdf,sampler);
                bsdf = pdf>0 ? eval(xi) : .0f;
            }
            else {
                // SIR when multiple edges are present to improve IS quality
                static constexpr std::size_t N = 8;
                Float bsdfs[N];
                Float aggw[N];
                Vector2 xis[N];

                Float sumw=.0f;
                for (std::size_t n=0;n<N;++n) {
                    const auto rand = sampler->next1D() * sumPhat_j;
                    const auto& edgeit = std::lower_bound(edges.begin(), edges.end(), rand,
                                                        [](const auto& e, auto v) {
                                                                return e.PhatAccum<v;
                                                        });
                    const auto& e = edgeit!=edges.end() ? *edgeit : edges.back();

                    Float pdf;
                    xis[n] = sampleEdge(e,pdf,sampler);
                    bsdfs[n] = pdf>0 ? eval(xis[n]) : .0f;
                    const auto w = pdf>0 ? bsdfs[n]/pdf : .0f;
                    sumw += w;
                    aggw[n] = sumw;
                }
                const auto n = std::min<std::size_t>(N-1,
                                    std::lower_bound(&aggw[0], &aggw[N], sampler->next1D()*sumw)-(&aggw[0])
                               );
                xi = xis[n];
                bsdf = bsdfs[n];
                pdf = sumw>0 ? Float(N)*bsdf/sumw : .0f;
            }

            if (pdf>0) {
                // xi in exit frame
                wo = -Vector3{ xi.x,xi.y,sqrtf(std::max<Float>(0,1-dot(xi,xi))) };
                wo = frame.toWorld(wo);

                return bsdf/pdf;
            }

            wo = {};
            return {};
        }

        Vector3 findExitPoint(Sampler *sampler) const {
            const auto rand = sampler->next1D() * sumPhat_j;
            const auto& edgeit = std::lower_bound(edges.begin(), edges.end(), rand,
                                                  [](const auto& e, auto v) {
                                                        return e.PhatAccum<v;
                                                  });
            return findExitPoint(edgeit, sampler);
        }

        Float P() const { return Ppl_A; }

        Float pdf(const Vector3 &wi, const Vector3 &wo) const {
            const auto wolocal = frame.toLocal(wo);
            if (wolocal.z>=0) return 0;
            const auto xi = -Vector2{ wolocal.x,wolocal.y };
            return evalPdf(xi);
        }

        Float bsdf(const Vector3 &wo) const {
            const auto wolocal = frame.toLocal(wo);
            if (wolocal.z>=0) return 0;
            const auto xi = -Vector2{ wolocal.x,wolocal.y };
            return eval(xi);
        }

        static fsdPrecomputedTables tables;

        Frame frame;
        Float Ppl_A=.0f, sumPhat_j, Ppl_0, k, beam_sigma;
        std::vector<edge_t> edges;

    private:
        Vector2 sampleEdge(const edge_t &e, Float &pdf, Sampler *sampler) const {
            const Matrix2x2 Xi = k*Matrix2x2(e.e.x,e.e.y,e.e.y,-e.e.x);
            Matrix2x2 invXi = { 0,0,0,0 };
            Xi.invert2x2(invXi);

            const float A = std::norm(e.a-e.b), B = std::norm(e.a+e.b)/4.f;
            const bool alpha1 = sampler->next1D()*(A+B) <= A;
            const auto rand3 = Point3{ sampler->next1D(),sampler->next1D(),sampler->next1D() };
            const auto xi = invXi * (alpha1 ? 
                          tables.importanceSampleCDF1(rand3) : 
                          tables.importanceSampleCDF2(rand3));

            // PDF
            pdf = evalPdf(xi);
            return xi;
        }

        Vector3 findExitPoint(const decltype(fsdBsdf::edges)::const_iterator &eit, Sampler *sampler) const {
            const auto search_radius = get_search_radius();

            // Edge-adjacent point
            if (eit!=edges.end()) {
                const auto m = Vector2{ eit->e.y,-eit->e.x };
                const auto a = Point2{ eit->v-.5f*eit->e };
                const auto b = Point2{ eit->v+.5f*eit->e };

                float t1,t2;
                intersectCircleLine(search_radius, a,b, t1,t2);
                const auto p1 = a + std::min(1.f,std::max(.0f,t1))*(b-a);
                const auto p2 = a + std::min(1.f,std::max(.0f,t2))*(b-a);
                // Random point on edge
                auto ep = p1 + sampler->next1D()*(p2-p1);
                // Random offset
                const auto lm = m.length();
                const auto minofst = lm*1.f/100.f;
                const auto maxofst = lm*1.f/4.f;
                const auto max = std::min(maxofst, std::max(.0f,2*search_radius-Vector2{ ep }.length()));
                const auto min = std::min(max,minofst);

                const auto rand = sampler->next1D()*(max - min) + min;
                ep += rand*m/lm;

                return frame.toWorld(Vector3{ ep.x,ep.y,-search_radius/2 });
            }

            return {0,0,0};
        }
    };

    const fsdBsdf constructFSDBSDF(Frame frame, Point3 p, Vector3 wi, const Scene* scene, int &spec_idx) const {
        spec_idx = 0;
        const float wavelength = Spectrum::m_wavelengths[spec_idx];
        const float sigma = m_sigma*wavelength;
        const auto k = 2*M_PI/wavelength;

        using cache_key = std::pair<Point3,Vector3>;
        struct cmp {
            bool operator() (const cache_key& a, const cache_key& b) const {
                return a.first.x!=b.first.x ? a.first.x<b.first.x : 
                       a.first.y!=b.first.y ? a.first.y<b.first.y :
                       a.first.z!=b.first.z ? a.first.z<b.first.z :
                       a.second.x!=b.second.x ? a.second.x<b.second.x : 
                       a.second.y!=b.second.y ? a.second.y<b.second.y :
                                                a.second.z<b.second.z;
            }
        };
        static thread_local std::map<cache_key, fsdBsdf, cmp> cache;
        if (cache.size()>100000)
            cache.clear();

        const auto key = cache_key{ p, wi };
        auto it = cache.lower_bound(key);
        if(it!=cache.end() && it->first==key) 
            return it->second;

        return cache.emplace_hint(it, key, 
                                  fsdBsdf{ scene, p, wi, frame, k, sigma }
                        )->second;
    }


public:
    Float getRoughness(const Intersection &its, int component) const {
        return m_bsdf->getRoughness(its, component);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "FreeSpaceDiffractionBSDF[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  bsdf = " << indent(m_bsdf->toString()) << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    bool m_enabled,m_SIR;
    ref<BSDF> m_bsdf;
    float m_sigma,m_scale;
};

fsdPrecomputedTables FreeSpaceDiffractionBSDF::fsdBsdf::tables;


MTS_IMPLEMENT_CLASS_S(FreeSpaceDiffractionBSDF, false, BSDF)
MTS_EXPORT_PLUGIN(FreeSpaceDiffractionBSDF, "FreeSpaceDiffraction BSDF")
MTS_NAMESPACE_END
