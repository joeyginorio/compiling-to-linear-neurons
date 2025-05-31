from dataclasses import dataclass
from collections.abc import Callable
from collections.abc import Mapping
from torch import nn, tensor, Tensor
import torch

# ------------------------ Types ------------------------ #
type Ty = TyBool | TyFun

@dataclass
class TyBool: ...

@dataclass
class TyFun: 
    ty1 : Ty
    ty2 : Ty

# ------------------------ Terms ------------------------ #
type Tm = TmVar | TmTrue | TmFalse | TmIf | TmFun | TmApp

@dataclass
class TmVar:
    name: str

@dataclass
class TmTrue: ...

@dataclass
class TmFalse: ...

@dataclass
class TmIf:
    tm1: Tm
    tm2: Tm
    tm3: Tm

@dataclass
class TmFun:
    name: str
    ty: Ty
    tm: Tm

@dataclass
class TmApp:
    tm1 : Tm
    tm2 : Tm

# ------------------------ Typing ------------------------ #
type Ctx = Mapping[str, Ty]

def typeOf(tm: Tm, ctx: Ctx) -> Ty:
    match tm:
        case TmVar(name):
            return ctx.pop(name)
        case TmTrue():
            return TyBool()
        case TmFalse():
            return TyBool()
        case TmFun(x, ty, tm):
            ctx.update({x : ty})
            return TyFun(ty, typeOf(tm, ctx))
        case TmIf(tm1, tm2, tm3):
            ty1 = typeOf(tm1, ctx)
            if ty1 == TyBool():
                ctx2 = ctx.copy()
                ctx3 = ctx.copy()
                ty2 = typeOf(tm2, ctx2)
                ty3 = typeOf(tm3, ctx3)
                ctx.clear()
                ctx.update(ctx2 | ctx3)
                if ty2 == ty3:
                    return ty3
                else:
                    raise TypeError("TmIf: Differing branch types.")
            else:
                raise TypeError("TmIf: Condition not type Bool.")
        case TmApp(tm1, tm2):
            ty1 = typeOf(tm1, ctx)
            ty2 = typeOf(tm2, ctx)
            match ty1:
                case TyBool():
                    raise TypeError("TmApp: LHS not of function type.")
                case TyFun(ty11, ty12):
                    if ty11 == ty2:
                        return ty12
                    else:
                        raise TypeError("TmApp: RHS doesn't match argument type.")
            

def check(tm: Tm, ctx: Ctx) -> Ty:
    print(f"Initial ctx: {ctx}")
    ty = typeOf(tm, ctx)
    print(f"Post ctx: {ctx}")
    if ctx == {}:
        return ty
    else:
        raise TypeError(f"Unused variables in the context: {ctx}")
    

# ------------------------ Typing ------------------------ #
type Cajal = tuple[Ctx, Tm, Ty]
type Env = Mapping[str, Tensor]

tt = TmTrue()
ff = TmFalse()
tyb = TyBool()

def compile(tm: Tm, env: Env, batch_size: int) -> Tensor:
    match tm:
        case TmVar(x):
            return env.get(x)
        case TmTrue():
            return tensor([[1.],[0.]]).unsqueeze(0).repeat(batch_size,1,1).to('mps')
        case TmFalse():
            return tensor([[0.],[1.]]).unsqueeze(0).repeat(batch_size,1,1).to('mps')
        case TmIf(tm1, tm2, tm3):
            b = compile(tm1, env, batch_size)
            v2 = compile(tm2, env, batch_size)
            v3 = compile(tm3, env, batch_size)
            # print("b:", b.shape)
            # print("v2:", v2.shape)
            # print("v3:", v3.shape)
            # safe, contiguous selectors
            b0 = b[:, 0].view(-1, 1, 1)     # [B,1,1]
            b1 = b[:, 1].view(-1, 1, 1)
            # print("b0:", b0.shape)
            # print("b1:", b1.shape)
            # return b[:,0].view(-1,1,1) * v2 + b[:,1].view(-1,1,1) * v3
            return b0 * v2 + b1 * v3
        case TmFun(x, ty, tm):
            outs = []
            for b in basis(ty, batch_size):
                outs.append(compile(tm, {x : b}, batch_size))
            return torch.cat(outs, dim=2)
        case TmApp(tm1, tm2):
            v1 = compile(tm1, env, batch_size)
            v2 = compile(tm2, env, batch_size).view(batch_size,-1, 1)
            return v1 @ v2


def basis(ty: Ty, batch_size) -> tensor:
    match ty:
        case TyBool():
            bs = []
            for i in range(2):
                bs_ = torch.eye(2)
                bs.append(bs_[:, i: i+1].unsqueeze(0).expand(4, 2, -1))
            return bs
        case TyFun(ty1, ty2):
            in_dim = dim(ty1)
            out_dim = dim(ty2)
            bs = []
            for i in range(out_dim):
                for j in range(in_dim):
                    bs_ = torch.zeros(out_dim, in_dim)
                    bs_[i,j] = 1.
                    bs.append(bs_)
            return bs


def dim(ty: Ty) -> int:
    match ty:
        case TyBool():
            return 2
        case TyFun(ty1, ty2):
            return dim(ty1) * dim(ty2)

# ------------------------ Tests ------------------------ #

# Should pass
def test1():
    ite1 = TmIf(TmVar('x'), TmVar('y'), TmVar('y'))
    ctx1 = {"x": TyBool(), "y": TyBool()}
    print(check(ite1, ctx1))

# Should fail
def test2():
    ite1 = TmIf(TmVar('x'), TmVar('y'), TmVar('y'))
    ctx1 = {"x": TyBool(), "y": TyBool(), "z" : TyBool()}
    print(check(ite1, ctx1))

# Should pass
def test3():
    ite1 = TmFun("x", TyBool(), TmVar('x'))
    ctx1 = {}
    print(check(ite1, ctx1))

# Should fail
def test4():
    ite1 = TmFun("x", TyBool(), TmVar('y'))
    ctx1 = {'y' : TyBool()}
    print(check(ite1, ctx1))

# Should pass
def test5():
    tm1 = TmFun("x", TyBool(), TmVar('x'))
    tm2 = TmTrue() 
    ctx1 = {}
    print(check(TmApp(tm1, tm2), ctx1))

