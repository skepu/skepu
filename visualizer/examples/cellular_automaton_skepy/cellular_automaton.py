# Cellular automaton (1D rule table and Game of Life in 2D) with MapOverlap, Region1D and
# Region2D, edge mode Pad, a Mat proxy as rule table and Map with a Random generator
# (translation of SkePU's examples/cellular_automaton.cpp; the trace calls are no-ops unless
# built with shedskin --trace).
# Usage: cellular_automaton DIM SIZE ITERATIONS INIT-ALIVE-RATIO BACKEND
import skepu
import sys

def init_random_f(random, alive_ratio):
    r = random.get_normalized()
    if r < alive_ratio:
        return 1
    return 0

def automaton1d(r, rules):
    newval = 0
    for i in range(-r.oi, r.oi + 1):
        if i != 0:
            if r[i]:
                newval += 1
    if rules[r[0], newval]:
        return 1
    return 0

def automaton2d(r, rules):
    newval = 0
    for i in range(-r.oi, r.oi + 1):
        for j in range(-r.oj, r.oj + 1):
            if not (i == 0 and j == 0):
                if r[i, j]:
                    newval += 1
    if rules[r[0, 0], newval]:
        return 1
    return 0

def life1d(size, iters, alive_ratio):
    init_random = skepu.Map(init_random_f)
    init_random.set_prng(skepu.PRNG())
    update = skepu.MapOverlap(automaton1d)
    update.set_overlap(1)
    update.set_edge_mode(skepu.EDGE_Pad)
    update.set_pad(0)

    domain_a = skepu.Vector(size, 0)
    domain_b = skepu.Vector(size, 0)
    rules = skepu.Matrix(2, 3, 0)
    with skepu.external(write=rules):
        rules[0, 1] = 1
    skepu.virtual_dataset("Domain", domain_a, domain_b)

    init_random(domain_a, rand=skepu.Random(1), uniforms=alive_ratio)
    skepu.snapshot(domain_a)
    for i in range(0, iters, 2):
        with skepu.scope("Double-iteration"):
            update(domain_b, domain_a, proxies=rules)
            update(domain_a, domain_b, proxies=rules)
            skepu.snapshot(domain_b)
            skepu.snapshot(domain_a)
    print(domain_a)

def life2d(size, iters, alive_ratio):
    init_random = skepu.Map(init_random_f)
    init_random.set_prng(skepu.PRNG())
    update = skepu.MapOverlap(automaton2d)
    update.set_overlap(1, 1)
    update.set_edge_mode(skepu.EDGE_Pad)
    update.set_pad(0)

    domain_a = skepu.Matrix(size, size, 0)
    domain_b = skepu.Matrix(size, size, 0)
    rules = skepu.Matrix(2, 9, 0)
    with skepu.external(write=rules):
        # Each empty cell with three neighbors becomes populated.
        rules[0, 3] = 1
        # Each populated cell with two or three neighbors survives.
        rules[1, 2] = 1
        rules[1, 3] = 1
    skepu.virtual_dataset("Domain", domain_a, domain_b)

    init_random(domain_a, rand=skepu.Random(1), uniforms=alive_ratio)
    skepu.snapshot(domain_a)
    for i in range(iters):
        with skepu.scope("Double-iteration"):
            update(domain_b, domain_a, proxies=rules)
            update(domain_a, domain_b, proxies=rules)
            skepu.snapshot(domain_b)
            skepu.snapshot(domain_a)
    print(domain_a)

def main():
    if len(sys.argv) < 6:
        print("Usage:", sys.argv[0], "dim size iterations init-alive-ratio backend")
        sys.exit(1)
    dim = int(sys.argv[1])
    size = int(sys.argv[2])
    iters = int(sys.argv[3])
    alive_ratio = float(sys.argv[4])
    spec = skepu.BackendSpec(sys.argv[5])
    skepu.set_global_backend_spec(spec)

    if dim == 1:
        life1d(size, iters, alive_ratio)
    if dim == 2:
        life2d(size, iters, alive_ratio)

main()
