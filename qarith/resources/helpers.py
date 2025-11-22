def getGateCount(circ):
    raw_counts = circ.count_ops()
    ops = {}
    for k, v in raw_counts.items():
        if isinstance(k, str):
            key = k
        else:
            # Instruction-like keys typically expose a 'name' attribute
            key = getattr(k, "name", str(k))
        ops[key] = v

    num_ccx = ops.get("ccx", 0)
    num_cx = ops.get("cx", 0)
    return num_ccx, num_cx