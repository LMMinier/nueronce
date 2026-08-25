"""Deterministic math/science verification primitives."""
import math

def arithmetic(spec):
    op,args=spec
    if op=="quadratic_roots":
        a,b,c=args; d=b*b-4*a*c
        return tuple(round(x,8) for x in ((-b+math.sqrt(d))/(2*a),(-b-math.sqrt(d))/(2*a)))
    if op=="dot":
        a,b=args; return sum(x*y for x,y in zip(a,b))
    if op=="mean": return sum(args)/len(args)
    raise KeyError(op)

def physics(spec):
    op,args=spec
    if op=="newton2": m,a=args; return m*a
    if op=="kinetic_energy": m,v=args; return .5*m*v*v
    if op=="ohms_law_v": i,r=args; return i*r
    if op=="wave_speed": f,wavelength=args; return f*wavelength
    raise KeyError(op)

def chemistry(spec):
    op,args=spec
    if op=="moles": mass,molar_mass=args; return mass/molar_mass
    if op=="ph_from_h":
        (h,)=args; return round(-math.log10(h),8)
    raise KeyError(op)
