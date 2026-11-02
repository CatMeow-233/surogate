import argparse,json,torch
from torch import nn
def expert(d=12,m=24):
    return nn.Sequential(nn.Linear(d,m,bias=False),nn.Tanh(),nn.Linear(m,d,bias=False))
def train_expert(seed,lang,steps=300):
    torch.manual_seed(seed)
    model=expert()
    opt=torch.optim.AdamW(model.parameters(),lr=.008)
    x=torch.randn(256,12)
    y=torch.tanh(x @ (A if lang==0 else B))
    for _ in range(steps):
        opt.zero_grad();loss=(model(x)-y).square().mean();loss.backward();opt.step()
    return model
def block_concat(e0,e1,x):
    # tanh MLP block concat exactly equals mean of two expert outputs
    W=torch.cat([e0[0].weight,e1[0].weight],dim=0)
    V=torch.cat([e0[2].weight,e1[2].weight],dim=1)/2
    return torch.tanh(x @ W.T) @ V.T
def run(seed,steps):
    global A,B
    torch.manual_seed(100+seed)
    A=torch.randn(12,12)*.2
    B=torch.randn(12,12)*.2
    e0=train_expert(10+seed,0,steps)
    e1=train_expert(20+seed,1,steps)
    torch.manual_seed(400+seed)
    x=torch.randn(512,12)
    z=e0(x).detach();en=e1(x).detach()
    oracle=torch.cat([z[:256],en[256:]])
    hard=torch.cat([z[:256],en[256:]])
    wrong=torch.cat([en[:256],z[256:]])
    avg=(z+en)/2
    # Direct parameter average is a distinct operation
    merged=expert()
    with torch.no_grad():
        for p,p0,p1 in zip(merged.parameters(),e0.parameters(),e1.parameters()):
            p.copy_((p0+p1)/2)
    merged_out=merged(x)
    concat=block_concat(e0,e1,x)
    assert (concat-avg).abs().max().item()<1e-5
    def mse(v):return float((v-oracle).square().mean())
    # single-branch output calibration: fixed expert, learned small bias, synthetic demonstration only
    bias=nn.Parameter(torch.zeros(2,12))
    opt=torch.optim.Adam([bias],lr=.03)
    for _ in range(80):
        out=torch.cat([z[:256]+bias[0],en[256:]+bias[1]])
        loss=(out-oracle).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    return {"seed":seed,"oracle":mse(oracle),"hard_top1":mse(hard),
      "wrong_route":mse(wrong),"weight_average":mse(merged_out),
      "top2_average":mse(avg),"2n_concat":mse(concat),
      "2n_top2_max_abs_diff":float((concat-avg).abs().max()),
      "single_route_bias_calibration":mse(torch.cat([z[:256]+bias[0],en[256:]+bias[1]]))}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--steps",type=int,default=300);p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2]);p.add_argument("--out",default="results.json");a=p.parse_args()
    rows=[run(s,a.steps) for s in a.seeds]
    with open(a.out,"w") as f:json.dump({"scope":"synthetic unit comparison, not language model result","results":rows},f,indent=2)
    print(json.dumps(rows,indent=2))
