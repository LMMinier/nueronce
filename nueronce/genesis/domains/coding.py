"""Coding-domain verifiers and initial skill curriculum."""
import ast

SAFE={"sum":sum,"range":range,"sorted":sorted,"len":len,"min":min,"max":max,
      "enumerate":enumerate,"zip":zip,"set":set,"list":list,"dict":dict}

def safe_expr(code: str):
    tree=ast.parse(code,mode="eval")
    forbidden=(ast.Import,ast.ImportFrom,ast.Lambda,ast.Await,ast.Yield,ast.NamedExpr)
    if any(isinstance(n,forbidden) for n in ast.walk(tree)):
        raise ValueError("forbidden syntax")
    return eval(compile(tree,"<genesis-code>","eval"),{"__builtins__":{}},SAFE)

def algorithm_case(spec):
    name,data=spec
    if name=="binary_search":
        arr,target=data; lo,hi=0,len(arr)-1
        while lo<=hi:
            mid=(lo+hi)//2
            if arr[mid]==target:return mid
            if arr[mid]<target:lo=mid+1
            else:hi=mid-1
        return -1
    if name=="dedupe_order":
        out=[];seen=set()
        for x in data:
            if x not in seen: seen.add(x);out.append(x)
        return out
    raise KeyError(name)
