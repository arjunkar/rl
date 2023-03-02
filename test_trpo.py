import torch

def test_cg():

    def cg(H, g):
        # Computes x = H^{-1}g by conjugate gradient
        def compute_Hvec(x):
            return H@x

        def sub(v1,v2):
            return v1-v2

        def add(v1,v2):
            return v1+v2

        def mult(scalar, v):
            return scalar*v

        def inner(v1,v2):
            return v1@v2

        x = torch.zeros_like(g)
        Hx = compute_Hvec(x)
        d = sub(g, Hx)
        r = sub(g, Hx)

        # Run conjugate gradient algorithm, updating x until close to H^{-1}g
        # or out of iterations
        iter = 0
        while iter < H.size()[0]:
            # Debugging purposes
            objective = inner(x, Hx) - 2*inner(g, x)

            beta_denom = inner(r, r)
            Hd = compute_Hvec(d)

            alpha = beta_denom / inner(d, Hd)

            x = add(x, mult(alpha, d))
            r = sub(r, mult(alpha, Hd))
            beta = inner(r, r) / beta_denom
            d = add(r, mult(beta, d))

            Hx = compute_Hvec(x)
            iter += 1
        return x

    def build_Hg(n):
        H = (n*torch.randn(size=(n,n))).abs().round()
        H = H @ H.transpose(0,1) + 1.0
        g = (n*torch.randn(n)).round()
        return H.to(dtype=torch.float64), g.to(dtype=torch.float64)
    
    def cg_tester(n):
        H, g = build_Hg(n)
        x_cg = cg(H, g)
        x_inv = torch.inverse(H) @ g
        assert(
        torch.allclose(x_cg, x_inv)
        )

    cg_tester(2)
    cg_tester(3)
    cg_tester(5)
    cg_tester(8)
    # Due to the buildup of floating point errors in the
    # conjugate gradient method, it is difficult to test this
    # function for matrices larger than 10x10.
    # torch.float64 is sufficient to test 8x8 matrices.
    
def test_sec_deriv():

    a = torch.tensor(3., requires_grad=True)
    b = torch.tensor(5., requires_grad=True)
    potential = (a*b)**2 + a**4 + b**3
    
    H = torch.tensor(
        [
            [12*a**2 + 2*b**2, 4*a*b],
            [4*a*b, 2*a**2 + 6*b]
        ]
    )

    def vecGen():
        return [torch.randn(1)[0], torch.randn(1)[0]]

    # First derivative of potential
    force = torch.autograd.grad(
                    potential, 
                    [a,b],
                    create_graph=True # Allow for second derivative
                )

    # Use torch.autograd.grad to double-differentiate
    def compute_Hvec(vec):
        return torch.autograd.grad(
                force,
                [a,b],
                grad_outputs=vec,
                retain_graph=True
            )

    def hess_tester():
        v = vecGen()
        Hv = compute_Hvec(v)
        Htv = H @ torch.tensor(v)
        assert(torch.allclose( torch.tensor(Hv) , Htv))

    hess_tester()
    hess_tester()
    hess_tester()
    hess_tester()
    hess_tester()