"""Reproduz as análises da dissertação; isolado e retomável."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
OUT=Path('results/dissertation');FIG=Path('figures/dissertation')
OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str((OUT/'cache/matplotlib').resolve());os.environ['XDG_CACHE_HOME']=str((OUT/'cache').resolve())
import argparse,subprocess,sys
from dpmeta import dissertation_selector as sel
from scripts import analyze_dissertation_extras as x
from scripts import regen_tied_winner_figures as tw
from scripts import dissertation_analysis as a
CONDITIONS='results/portfolio-984-sgd/conditions_15.csv'
LODO='results/experiments/lodo_gbm15/lodo_picks.csv';MANIFEST='data/openml_manifest_984_completed.csv'
LABELS={sel.NB8U:'Laplace · NB b8u','softmax__dp_sgd_clip2_lamrule':'DP-SGD · Softmax (regra λ)',
        'softmax__dp_sgd_clip2_lam0003':'DP-SGD · Softmax (λ=0,0003)'}
x.SHORT.update(LABELS);tw.PT.update(LABELS);tw.x.SHORT.update(LABELS)

def log(step):
    with Path('../mestrado-papers/V5_1_CHANGELOG.md').open('a') as f:f.write('\n- Etapa 1: concluído `'+step+'` pelo gerador isolado final.\n')

def run(name):subprocess.run([sys.executable,'scripts/'+name+'.py'],check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='all');args=parser.parse_args()
    stages=['extras','modes','tables','learning','publication','documents','build','validate'] if args.stage=='all' else [args.stage]
    for stage in stages:
        flag=OUT/(stage+'.done')
        if flag.exists():print('REUSE',stage,flush=True);continue
        if stage=='extras':
            x.main(['--selector','gbm_residual_fallback','--conditions',CONDITIONS,'--lodo',LODO,
                    '--output',str(OUT/'extras'),'--figures',str(FIG),'--seed',str(sel.SEED)])
        elif stage=='modes':a.modes_analysis(CONDITIONS,MANIFEST,OUT/'modes')
        elif stage=='tables':
            a.base(CONDITIONS,MANIFEST,OUT,LODO)
            for name in ['analyze_supplementary','analyze_portfolio_common','analyze_rules']:run(name)
            subprocess.run([sys.executable,'scripts/analyze_noise.py','--conditions',CONDITIONS],check=True)
            a.size_analysis(OUT/'outer_predictions.csv',MANIFEST,OUT/'size',CONDITIONS)
        elif stage=='learning':
            from concurrent.futures import ThreadPoolExecutor
            def fold_job(fold):
                path=OUT/f'learning_fold{fold}.log'
                with path.open('w') as stream:
                    subprocess.run([sys.executable,'scripts/analyze_learning_curve.py','--fold',str(fold)],stdout=stream,stderr=subprocess.STDOUT,check=True)
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fold_job,range(5)))
            run('analyze_learning_curve');run('analyze_epsilon_strategy')
        elif stage=='publication':
            a.publication(CONDITIONS,MANIFEST,OUT,FIG,LODO);run('finalize_dissertation')
        elif stage=='documents':
            run('rewrite_documents')
            from scripts.finalize_dissertation import numbers
            numbers()
        elif stage=='build':run('build_and_check')
        elif stage=='validate':run('validate_dissertation')
        else:raise ValueError(stage)
        flag.write_text('complete\n');log(stage)
if __name__=='__main__':main()
