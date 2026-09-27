import argparse, json, logging, math, os, random, platform, sys, time, hashlib
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def dft_codebook(n, nb):
    theta = np.linspace(0, 2*np.pi, nb, endpoint=False)
    return np.exp(1j*np.arange(n)[None,:]*theta[:,None])/np.sqrt(n)

def zadoff_chu_codebook(n, nb):
    """ZC-derived candidate codebook. For small composite n, cyclic shifts are limited;
    candidates are deduplicated up to global phase and repeated only if necessary."""
    roots = [q for q in range(1, n) if math.gcd(q, n) == 1]
    cand = []
    base_n = np.arange(n)
    for q in roots:
        if n % 2:
            z = np.exp(-1j*np.pi*q*base_n*(base_n+1)/n)
        else:
            z = np.exp(-1j*np.pi*q*base_n**2/n)
        for shift in range(n):
            cand.append(np.roll(z, shift)/np.sqrt(n))
    # Remove global-phase duplicates.
    uniq = []
    keys = set()
    for v in cand:
        k0 = np.flatnonzero(np.abs(v) > 1e-10)[0]
        v2 = v * np.exp(-1j*np.angle(v[k0]))
        key = tuple(np.round(np.c_[v2.real, v2.imag], 10).ravel())
        if key not in keys:
            keys.add(key); uniq.append(v)
    if not uniq:
        uniq = [np.ones(n, dtype=complex)/np.sqrt(n)]
    out = np.asarray(uniq)
    if len(out) < nb:
        # Never invent new spatial directions: pad with DFT beams so validation exposes the hybrid.
        d = dft_codebook(n, nb)
        out = np.vstack([out, d])[:nb]
    return out[:nb]

def codebook(name, n, nb):
    if name.lower() == 'dft': return dft_codebook(n, nb)
    if name.lower() in ('zadoff-chu','zc','zadoff_chu'): return zadoff_chu_codebook(n, nb)
    if name.lower() == 'random':
        x = np.random.randn(nb,n)+1j*np.random.randn(nb,n)
        return x/np.linalg.norm(x,axis=1,keepdims=True)
    raise ValueError(f'Unknown codebook: {name}')

def normalize(h):
    h=np.asarray(h,dtype=complex); return h/(np.linalg.norm(h)+1e-12)

def steering(n, angle):
    return np.exp(1j*np.pi*np.arange(n)*np.sin(angle))/np.sqrt(n)

def channel_3gpp_uma(n, frequency_ghz=28.0, los_prob=0.5):
    """3GPP TR 38.901-inspired UMa surrogate: clustered geometric channel.
    This is deliberately labeled a surrogate, not a standards-complete implementation."""
    los = np.random.rand() < los_prob
    h = np.zeros(n, dtype=complex)
    if los:
        aoa = np.random.uniform(-np.pi/2, np.pi/2)
        # Frequency-dependent phase term; wavelength scaling is implicit in normalized ULA.
        h += np.sqrt(10**(8/10)/(10**(8/10)+1))*steering(n, aoa)
    k_db = 8 if los else -np.inf
    k_lin = 10**(k_db/10) if np.isfinite(k_db) else 0.0
    clusters = 12 if los else 19
    rays = 20
    for _ in range(clusters):
        c_gain=(np.random.randn()+1j*np.random.randn())/np.sqrt(2*clusters)
        center=np.random.uniform(-np.pi/2,np.pi/2)
        spread=np.deg2rad(5 if los else 10)
        for _ in range(rays):
            aoa=np.clip(center+np.random.laplace(0,spread/np.sqrt(2)),-np.pi/2,np.pi/2)
            phase=np.exp(-1j*2*np.pi*np.random.rand())
            h += c_gain*phase*steering(n,aoa)/np.sqrt(rays)
    if k_lin>0:
        # keep LoS component and clustered component at a controlled K-factor
        h = np.sqrt(k_lin/(k_lin+1))*normalize(h) + np.sqrt(1/(k_lin+1))*h
    return normalize(h)

def channel_rayleigh(n):
    return normalize((np.random.randn(n)+1j*np.random.randn(n))/np.sqrt(2))

def channel_rician(n, K=5):
    los=steering(n,np.random.uniform(-np.pi/2,np.pi/2))
    nlos=(np.random.randn(n)+1j*np.random.randn(n))/np.sqrt(2)
    return normalize(np.sqrt(K/(K+1))*los+np.sqrt(1/(K+1))*nlos)

def load_deepmimo(path, n):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f'DeepMIMO channel file not found: {p}')
    if p.suffix.lower()=='.npy':
        arr=np.load(p)
    elif p.suffix.lower()=='.npz':
        z=np.load(p)
        arr=z['H'] if 'H' in z else z[list(z.keys())[0]]
    else:
        raise ValueError('DeepMIMO input must be .npy or .npz containing complex channel samples')
    arr=np.asarray(arr)
    if arr.ndim==1: arr=arr[None,:]
    if arr.ndim>2: arr=arr.reshape(arr.shape[0],-1)
    if np.iscomplexobj(arr): h=arr
    elif arr.shape[1]==2*n: h=arr[:,:n]+1j*arr[:,n:]
    else: raise ValueError(f'Cannot map DeepMIMO array shape {arr.shape} to {n} antennas')
    if h.shape[1]!=n: raise ValueError(f'DeepMIMO samples have {h.shape[1]} elements, expected {n}')
    return np.asarray([normalize(x) for x in h])

def make_channel(cfg, rng_state=None, deepmimo=None):
    n=cfg['num_antennas']; model=cfg.get('channel_model','rayleigh').lower()
    if model=='rayleigh': return channel_rayleigh(n)
    if model=='rician': return channel_rician(n,cfg.get('rician_k',5))
    if model in ('3gpp-38.901-uma','3gpp_uma','3gpp-uma','3gpp_38_901_uma'):
        return channel_3gpp_uma(n,cfg.get('frequency_ghz',28.0),cfg.get('los_probability',0.5))
    if model=='deepmimo':
        if deepmimo is None: raise RuntimeError('DeepMIMO samples were not loaded')
        return deepmimo[np.random.randint(len(deepmimo))]
    raise ValueError(f'Unknown channel model: {model}')

def imperfect_estimate(h, quality):
    quality=float(np.clip(quality,0,1))
    e=channel_rayleigh(len(h))
    return normalize(np.sqrt(quality)*h + np.sqrt(max(1-quality,0))*e)

def metrics(h,w,snr_db,bw=10.0):
    p=float(abs(np.vdot(h,w))**2); snr=10**(snr_db/10)
    rate=float(np.log2(1+snr*p)); ber=0.5*math.erfc(math.sqrt(max(snr*p,1e-12)))
    return rate,bw*rate,ber,p

class Net(nn.Module):
    def __init__(self,n,nb,algo):
        super().__init__(); self.algo=algo
        self.fc1=nn.Linear(2*n,64); self.fc2=nn.Linear(64,32)
        if algo=='Dueling DQN': self.v=nn.Linear(32,1); self.a=nn.Linear(32,nb)
        else: self.out=nn.Linear(32,nb)
    def forward(self,x):
        x=torch.relu(self.fc1(x)); x=torch.relu(self.fc2(x))
        if self.algo=='Dueling DQN':
            a=self.a(x); return self.v(x)+a-a.mean(dim=1,keepdim=True)
        return self.out(x)

class Agent:
    def __init__(self,n,nb,algo,cfg):
        self.nb=nb; self.gamma=cfg['gamma']; self.eps=1.; self.eps_min=.01
        self.eps_decay=cfg['epsilon_decay']; self.net=Net(n,nb,algo); self.target=Net(n,nb,algo)
        self.target.load_state_dict(self.net.state_dict())
        self.opt=optim.Adam(self.net.parameters(),lr=cfg['learning_rate']); self.loss=nn.SmoothL1Loss()
        self.mem=deque(maxlen=5000)
    def act(self,s,train=True):
        if train and random.random()<self.eps: return random.randrange(self.nb)
        with torch.no_grad(): return int(torch.argmax(self.net(torch.tensor(s,dtype=torch.float32)[None,:])).item())
    def step(self,batch):
        if len(self.mem)<batch: return None
        b=random.sample(self.mem,batch); s,a,r,ns,d=zip(*b)
        s=torch.tensor(np.array(s),dtype=torch.float32); ns=torch.tensor(np.array(ns),dtype=torch.float32)
        a=torch.tensor(a,dtype=torch.long); r=torch.tensor(r,dtype=torch.float32); d=torch.tensor(d,dtype=torch.float32)
        q=self.net(s).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad():
            if self.net.algo in ('Double DQN','Dueling DQN'):
                na=self.net(ns).argmax(1); nq=self.target(ns).gather(1,na[:,None]).squeeze(1)
            else: nq=self.target(ns).max(1).values
            y=r+self.gamma*(1-d)*nq
        loss=self.loss(q,y); self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.eps=max(self.eps*self.eps_decay,self.eps_min); return float(loss.item())
    def remember(self,*x): self.mem.append(x)
    def sync(self): self.target.load_state_dict(self.net.state_dict())

def state_from(h): return np.r_[h.real,h.imag].astype(np.float32)

def evaluate_agent(ag, cb, cfg, seed, condition, channel_pool=None):
    rows=[]; n=cfg['num_antennas']; delay=int(condition['feedback_delay_steps']); q=float(condition['csi_quality'])
    for snr_db in cfg['snr_grid']:
        ar=[]; ao=[]; rb=[]; ab=[]
        for i in range(cfg['eval_episodes']):
            if channel_pool is not None:
                h=channel_pool[np.random.randint(len(channel_pool))]
            else: h=make_channel(cfg,deepmimo=channel_pool)
            history=deque([h.copy() for _ in range(max(delay+1,1))],maxlen=delay+1)
            # Generate the delayed observation from past channel state.
            obs_true=history[0] if delay else h
            hhat=imperfect_estimate(obs_true,q)
            a=ag.act(state_from(hhat),False)
            r,_,b,_=metrics(h,cb[a],snr_db,cfg['bandwidth_mhz']); ar.append(r); rb.append(b)
            gains=[metrics(h,w,snr_db,cfg['bandwidth_mhz'])[0] for w in cb]; ao.append(max(gains))
            ab.append(0.5*math.erfc(math.sqrt(max(10**(snr_db/10)*max(abs(np.vdot(h,w))**2 for w in cb),1e-12))))
        rows.append({'seed':seed,'algorithm':condition['algorithm'],'channel_model':cfg.get('channel_model','rayleigh'),
                     'codebook':condition['codebook'],'gamma':condition['gamma'],'snr_db':snr_db,'csi_quality':q,
                     'feedback_delay_steps':delay,'agent_rate':np.mean(ar),'oracle_rate':np.mean(ao),
                     'agent_ber':np.mean(rb),'oracle_ber':np.mean(ab),'gap_bpshz':np.mean(ao)-np.mean(ar)})
    return pd.DataFrame(rows)

def run_one(algo,seed,cfg,logger,condition,deepmimo=None):
    seed_all(seed); n=cfg['num_antennas']; nb=cfg['num_beams']; cb=codebook(condition['codebook'],n,nb)
    train_cfg=dict(cfg); train_cfg['gamma']=condition['gamma']; ag=Agent(n,nb,algo,train_cfg)
    rows=[]; losses=[]; t0=time.time(); delay=int(condition['feedback_delay_steps']); q=float(condition['csi_quality'])
    history=deque(maxlen=max(delay+1,1)); h=make_channel(cfg,deepmimo=deepmimo); history.extend([h.copy()]*(delay+1))
    for ep in range(cfg['episodes']):
        total=0.; rr=[]; bb=[]; cc=[]
        # fresh episode channel, then evolve with a simple correlated perturbation
        h=make_channel(cfg,deepmimo=deepmimo); history.clear(); history.extend([h.copy()]*(delay+1))
        for _ in range(cfg['steps_per_episode']):
            obs=history[0] if delay else h; hhat=imperfect_estimate(obs,q)
            s=state_from(hhat); a=ag.act(s,True)
            rate,cap,ber,_=metrics(h,cb[a],cfg['snr_db'],cfg['bandwidth_mhz'])
            noise=(np.random.randn(n)+1j*np.random.randn(n))/np.sqrt(2)
            nh=normalize(h+cfg['mobility']*noise); history.append(nh.copy()); ns=state_from(imperfect_estimate(history[0] if delay else nh,q))
            ag.remember(s,a,rate,ns,False); loss=ag.step(cfg['batch_size'])
            h=nh; total+=rate; rr.append(rate); bb.append(ber); cc.append(cap)
            if loss is not None: losses.append(loss)
        if ep%cfg['target_update']==0: ag.sync()
        rows.append({'seed':seed,'algorithm':condition['algorithm'],'gamma':condition['gamma'],'codebook':condition['codebook'],
                     'csi_quality':q,'feedback_delay_steps':delay,'episode':ep+1,'reward':total,
                     'rate':rr[-1],'mean_rate':np.mean(rr),'ber':np.mean(bb),'capacity_mbps':np.mean(cc),
                     'epsilon':ag.eps,'loss':losses[-1] if losses else np.nan})
        if (ep+1)%max(1,cfg['episodes']//10)==0:
            logger.info('%s seed=%d gamma=%.2f cb=%s q=%.2f delay=%d episode=%d/%d last50=%.4f',
                        algo,seed,condition['gamma'],condition['codebook'],q,delay,ep+1,cfg['episodes'],np.mean([x['rate'] for x in rows[-50:]]))
    ev=evaluate_agent(ag,cb,cfg,seed,condition,channel_pool=deepmimo)
    logger.info('%s seed=%d completed in %.2fs',algo,seed,time.time()-t0)
    return pd.DataFrame(rows),ev

def t_ci(values, confidence=0.95):
    from scipy.stats import t
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if len(x)<2: return (float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan, len(x))
    m=float(np.mean(x)); se=float(np.std(x,ddof=1)/np.sqrt(len(x))); crit=float(t.ppf((1+confidence)/2,len(x)-1))
    return m,m-crit*se,m+crit*se,len(x)

def summarize_ci(ev):
    keys=['algorithm','gamma','codebook','channel_model','snr_db','csi_quality','feedback_delay_steps']
    rows=[]
    for key,g in ev.groupby(keys):
        vals=g.agent_rate.to_numpy(); m,lo,hi,n=t_ci(vals)
        _,dlo,dhi,_=t_ci(g.gap_bpshz.to_numpy())
        rows.append(dict(zip(keys,key if isinstance(key,tuple) else [key]),agent_rate_mean=m,agent_rate_ci_low=lo,
                          agent_rate_ci_high=hi,agent_rate_n=n,gap_mean=g.gap_bpshz.mean(),gap_ci_low=dlo,gap_ci_high=dhi))
    return pd.DataFrame(rows)

def paired_difference_ci(ev, reference_algorithm):
    keys=['gamma','codebook','channel_model','snr_db','csi_quality','feedback_delay_steps']
    rows=[]
    ref=ev[ev.algorithm==reference_algorithm]
    for algo,g in ev.groupby('algorithm'):
        if algo==reference_algorithm: continue
        m=g.merge(ref,on=['seed']+keys,suffixes=('_algo','_ref'))
        if m.empty: continue
        d=m.agent_rate_algo-m.agent_rate_ref
        mean,lo,hi,n=t_ci(d)
        rows.append({'algorithm':algo,'reference':reference_algorithm,**{k:m.iloc[0][k] for k in keys},
                     'paired_delta_rate_mean':mean,'paired_delta_ci_low':lo,'paired_delta_ci_high':hi,'n_seeds':n})
    return pd.DataFrame(rows)

def validate_codebooks(cfg,seeds):
    out=[]
    for cb_name in ['DFT','Zadoff-Chu']:
        for seed in seeds:
            seed_all(seed); cb=codebook(cb_name,cfg['num_antennas'],cfg['num_beams']); vals=[]
            for _ in range(cfg.get('codebook_validation_episodes',200)):
                h=make_channel(cfg); vals.append(max(abs(np.vdot(h,w))**2 for w in cb))
            out.append({'seed':seed,'codebook':cb_name,'mean_max_gain':np.mean(vals),'median_max_gain':np.median(vals)})
    df=pd.DataFrame(out)
    d=df[df.codebook=='DFT'].groupby('seed').mean(numeric_only=True)
    z=df[df.codebook=='Zadoff-Chu'].groupby('seed').mean(numeric_only=True)
    m=d.join(z,lsuffix='_dft',rsuffix='_zc')
    m['zc_to_dft_ratio']=m.mean_max_gain_zc/m.mean_max_gain_dft
    ratio=float(m.zc_to_dft_ratio.mean())
    decision='retain' if ratio >= cfg.get('zc_min_relative_gain',0.95) else 'drop'
    return df,m.reset_index(),decision,ratio

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/medium.json'); ap.add_argument('--out',default='results')
    ap.add_argument('--seeds',default=None); ap.add_argument('--quick',action='store_true')
    args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text()); out=Path(args.out); (out/'figures').mkdir(parents=True,exist_ok=True); (out/'tables').mkdir(parents=True,exist_ok=True); Path('logs').mkdir(exist_ok=True)
    seeds=[int(x) for x in (args.seeds or ','.join(map(str,cfg.get('seeds',range(42,52))))).split(',') if x.strip()]
    if args.quick: cfg['episodes']=min(cfg['episodes'],5); cfg['steps_per_episode']=min(cfg['steps_per_episode'],5); cfg['eval_episodes']=min(cfg['eval_episodes'],5)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.FileHandler('logs/experiment.log',mode='w'),logging.StreamHandler()])
    logger=logging.getLogger('experiment')
    logger.info('System: %s | Python %s | PyTorch %s',platform.platform(),sys.version.split()[0],torch.__version__)
    logger.info('Seeds: %s',seeds)
    deepmimo=None
    if cfg.get('channel_model','rayleigh').lower()=='deepmimo': deepmimo=load_deepmimo(cfg['deepmimo_path'],cfg['num_antennas'])
    # Codebook validation is independent of the learning algorithm.
    cb_raw,cb_sum,zc_decision,zc_ratio=validate_codebooks(cfg,seeds)
    cb_raw.to_csv(out/'tables'/'codebook_validation_by_seed.csv',index=False)
    cb_sum.to_csv(out/'tables'/'codebook_validation_summary.csv',index=False)
    logger.info('Zadoff-Chu validation: mean ZC/DFT oracle-gain ratio=%.4f -> %s',zc_ratio,zc_decision)
    conditions=[]
    for gamma in cfg.get('gammas',[cfg.get('gamma',0.95),0.0]):
        for q in cfg.get('csi_qualities',[1.0,0.8,0.5]):
            for delay in cfg.get('feedback_delay_steps',[0,1,3]):
                for cb in (['DFT'] if zc_decision=='drop' else cfg.get('codebooks_to_test',['DFT','Zadoff-Chu'])):
                    for algo in ['DQN','Double DQN','Dueling DQN'] + (['Contextual Bandit (gamma=0)'] if gamma == 0.0 else []):
                        conditions.append({'gamma':gamma,'csi_quality':q,'feedback_delay_steps':delay,'codebook':cb,'algorithm':algo})
    all_train=[]; all_eval=[]
    for cond in conditions:
        for seed in seeds:
            tr,ev=run_one(cond['algorithm'],seed,cfg,logger,cond,deepmimo); all_train.append(tr); all_eval.append(ev)
    train=pd.concat(all_train,ignore_index=True); ev=pd.concat(all_eval,ignore_index=True)
    train.to_csv(out/'tables'/'training_history_extended.csv',index=False); ev.to_csv(out/'tables'/'snr_results_extended.csv',index=False)
    ci=summarize_ci(ev); ci.to_csv(out/'tables'/'snr_summary_95ci.csv',index=False)
    diffs=paired_difference_ci(ev,'DQN'); diffs.to_csv(out/'tables'/'paired_algorithm_differences_95ci.csv',index=False)
    final=train.groupby(['algorithm','gamma','codebook','csi_quality','feedback_delay_steps','seed']).tail(cfg.get('final_window',50))
    final=final.groupby(['algorithm','gamma','codebook','csi_quality','feedback_delay_steps']).agg(
        final_rate_mean=('rate','mean'),final_rate_std=('rate','std'),final_ber_mean=('ber','mean'),
        final_capacity_mbps=('capacity_mbps','mean')).reset_index()
    final.to_csv(out/'tables'/'final_training_summary_extended.csv',index=False)
    # Compact figures for the primary 10-dB condition.
    primary=ci[(ci.snr_db==cfg['snr_db']) & (ci.codebook=='DFT') & (ci.gamma==cfg.get('gamma',0.95))]
    plt.figure(figsize=(9,5))
    for algo,g in primary.groupby('algorithm'):
        g=g.sort_values('feedback_delay_steps')
        plt.errorbar(g.feedback_delay_steps,g.agent_rate_mean,
                     yerr=[g.agent_rate_mean-g.agent_rate_ci_low,g.agent_rate_ci_high-g.agent_rate_mean],
                     marker='o',capsize=3,label=algo)
    plt.xlabel('Feedback delay (steps)'); plt.ylabel('Spectral efficiency (bit/s/Hz)')
    plt.title('Feedback-delay stress test at primary SNR'); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(out/'figures'/'feedback_delay_95ci.png',dpi=220); plt.close()
    qplot=primary[primary.feedback_delay_steps==0].sort_values('csi_quality')
    plt.figure(figsize=(9,5))
    for algo,g in qplot.groupby('algorithm'):
        g=g.sort_values('csi_quality')
        plt.errorbar(g.csi_quality,g.agent_rate_mean,
                     yerr=[g.agent_rate_mean-g.agent_rate_ci_low,g.agent_rate_ci_high-g.agent_rate_mean],
                     marker='o',capsize=3,label=algo)
    plt.xlabel('CSI quality (correlation parameter)'); plt.ylabel('Spectral efficiency (bit/s/Hz)')
    plt.title('Imperfect-CSI stress test at primary SNR'); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(out/'figures'/'imperfect_csi_95ci.png',dpi=220); plt.close()
    meta={'config':cfg,'seeds':seeds,'algorithms':['DQN','Double DQN','Dueling DQN','Contextual Bandit (gamma=0)'],
          'gamma_zero_contextual_bandit_baseline':True,'confidence_interval':'two-sided 95% Student-t across independent seeds',
          'paired_comparison':'same seed and condition matched against DQN',
          'codebook_validation':{'decision':zc_decision,'mean_zc_to_dft_oracle_gain_ratio':zc_ratio},
          'channel_note':'3GPP TR 38.901-inspired UMa surrogate or optional DeepMIMO .npy/.npz samples; neither is claimed to be a standards-complete channel simulator',
          'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'implementation':'PyTorch reproducibility runner with repeated seeds, imperfect CSI, feedback delay, gamma=0 baseline and codebook/channel validation'}
    (out/'run_metadata_extended.json').write_text(json.dumps(meta,indent=2))
    logger.info('Completed extended benchmark: %d seeds, %d condition rows',len(seeds),len(conditions))
if __name__=='__main__': main()
