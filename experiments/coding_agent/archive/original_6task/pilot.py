import os, re, json, csv, subprocess, tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from openai import OpenAI

MODEL = os.environ.get('PILOT_MODEL')
BASE_URL = os.environ.get('OPENAI_BASE_URL')
API_KEY = os.environ.get('OPENAI_API_KEY', 'dummy')

@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    prompt: str
    visible_tests: str
    hidden_tests: str

TASKS = [
    Task('dedupe','order', 'Implement dedupe(items) in solution.py. Remove duplicates while preserving first-occurrence order. Do not mutate input.',
         'from solution import dedupe\nassert dedupe([1,1,2,2,3]) == [1,2,3]\nassert dedupe(["a","b","a"]) == ["a","b"]\n',
         'from solution import dedupe\nassert dedupe([])==[]\nassert dedupe([3,2,3,1,2])==[3,2,1]\nx=[1,2,1]; b=list(x); assert dedupe(x)==[1,2]; assert x==b\n'),
    Task('first_unique','order', 'Implement first_unique(items) in solution.py. Return the first element occurring exactly once, else None.',
         'from solution import first_unique\nassert first_unique([1,2,1,3])==2\nassert first_unique(["a","a","b"])=="b"\n',
         'from solution import first_unique\nassert first_unique([]) is None\nassert first_unique([1,1,2,2]) is None\nassert first_unique(["x","y","x","z","y"])=="z"\n'),
    Task('stable_intersection','order', 'Implement stable_intersection(a,b) in solution.py. Return distinct values present in both, ordered by first occurrence in a.',
         'from solution import stable_intersection\nassert stable_intersection([1,2,3],[2,3,4])==[2,3]\n',
         'from solution import stable_intersection\nassert stable_intersection([], [1])==[]\nassert stable_intersection([3,1,3,2],[2,3])==[3,2]\n'),
    Task('chunks','boundaries', 'Implement chunks(seq,size) in solution.py. Consecutive chunks, final may be shorter. Raise ValueError if size<=0. Return list of lists.',
         'from solution import chunks\nassert chunks([1,2,3,4],2)==[[1,2],[3,4]]\nassert chunks([1,2,3],2)==[[1,2],[3]]\n',
         'from solution import chunks\nassert chunks([],3)==[]\nassert chunks([1],5)==[[1]]\ntry:\n chunks([1],0); raise AssertionError()\nexcept ValueError: pass\n'),
    Task('windows','boundaries', 'Implement sliding_windows(seq,width) in solution.py. Return all contiguous windows of exactly width. Raise ValueError if width<=0; width>len(seq) returns [].',
         'from solution import sliding_windows\nassert sliding_windows([1,2,3,4],2)==[[1,2],[2,3],[3,4]]\n',
         'from solution import sliding_windows\nassert sliding_windows([],1)==[]\nassert sliding_windows([1,2],3)==[]\ntry:\n sliding_windows([1],0); raise AssertionError()\nexcept ValueError: pass\n'),
    Task('clamp_slice','boundaries', 'Implement clamp_slice(seq,start,stop) in solution.py. Clamp endpoints to [0,len(seq)], interpret [start,stop), and return [] if clamped stop<start.',
         'from solution import clamp_slice\nassert clamp_slice([0,1,2,3],1,3)==[1,2]\nassert clamp_slice([0,1,2],-5,2)==[0,1]\n',
         'from solution import clamp_slice\nassert clamp_slice([0,1,2],2,99)==[2]\nassert clamp_slice([0,1,2],3,1)==[]\nassert clamp_slice([], -1, 9)==[]\n'),
]

INITIAL_SKILLS='- Follow the specification exactly.\n- Check edge cases before coding.\n'


def client():
    if not MODEL:
        raise RuntimeError('Set PILOT_MODEL first')
    kw={}
    if BASE_URL: kw['base_url']=BASE_URL
    return OpenAI(api_key=API_KEY, **kw)


def chat(messages):
    r=client().chat.completions.create(model=MODEL,messages=messages,temperature=0)
    return r.choices[0].message.content


def extract_code(text):
    m=re.search(r'```python\s*(.*?)```',text,re.S) or re.search(r'```\s*(.*?)```',text,re.S)
    return (m.group(1) if m else text).strip()+'\n'


def solve(task, skills):
    sys='You are a coding agent. Use persistent skills as advice, but current spec is authoritative. Return only complete solution.py in one python code block.'
    user=f'PERSISTENT SKILLS:\n{skills}\n\nTASK:\n{task.prompt}'
    return extract_code(chat([{'role':'system','content':sys},{'role':'user','content':user}]))


def update_memory(old, task, solution, verdict):
    sys='''Maintain a compact persistent coding playbook. You get previous skills, task, submitted solution, and ONLY proxy verdict PASS/FAIL. No hidden-test information. Keep <=10 markdown bullets. Extract reusable lessons. Treat PASS as evidence the approach may be useful; FAIL as evidence to revise/avoid it.'''
    user=f'''OLD SKILLS:\n{old}\n\nTASK:\n{task.prompt}\n\nSOLUTION:\n```python\n{solution}\n```\n\nPROXY VERDICT: {verdict}\n\nReturn updated markdown bullets only.'''
    return chat([{'role':'system','content':sys},{'role':'user','content':user}]).strip()


def run_tests(solution, tests):
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)
        (p/'solution.py').write_text(solution)
        (p/'test_task.py').write_text(tests)
        try:
            r=subprocess.run([os.sys.executable,'test_task.py'],cwd=p,capture_output=True,text=True,timeout=5)
            return r.returncode==0
        except subprocess.TimeoutExpired:
            return False


def text_distance(a,b):
    return 1-SequenceMatcher(None,a,b).ratio()


def run_trajectory(name, flip_round=None):
    out=Path('results')/name; out.mkdir(parents=True,exist_ok=True)
    skills=INITIAL_SKILLS; records=[]
    for t,task in enumerate(TASKS):
        print(f'[{name}] round {t}: {task.task_id}', flush=True)
        before=skills
        solution=solve(task,skills)
        proxy=run_tests(solution,task.visible_tests)
        oracle=run_tests(solution,task.hidden_tests)
        delivered=not proxy if t==flip_round else proxy
        skills=update_memory(before,task,solution,'PASS' if delivered else 'FAIL')
        rec=dict(round=t,task_id=task.task_id,family=task.family,proxy_pass=proxy,delivered_proxy_pass=delivered,oracle_pass=oracle,flipped=(t==flip_round),skills_before=before,skills_after=skills,solution=solution)
        records.append(rec)
        (out/f'{t:02d}_{task.task_id}.json').write_text(json.dumps(rec,indent=2))
    return records


def compare(clean, pert, flip_round):
    future=range(flip_round+1,len(clean))
    clean_oracle=[int(clean[t]['oracle_pass']) for t in future]
    pert_oracle=[int(pert[t]['oracle_pass']) for t in future]
    return {
        'flip_round': flip_round,
        'task_id': TASKS[flip_round].task_id,
        'family': TASKS[flip_round].family,
        'immediate_memory_distance': text_distance(clean[flip_round]['skills_after'],pert[flip_round]['skills_after']),
        'final_memory_distance': text_distance(clean[-1]['skills_after'],pert[-1]['skills_after']),
        'future_clean_oracle_rate': sum(clean_oracle)/len(clean_oracle),
        'future_pert_oracle_rate': sum(pert_oracle)/len(pert_oracle),
        'future_oracle_harm': sum(clean_oracle)/len(clean_oracle)-sum(pert_oracle)/len(pert_oracle),
        'future_solution_changes': sum(clean[t]['solution']!=pert[t]['solution'] for t in future),
        'future_tasks': len(list(future)),
    }


def main():
    Path('results').mkdir(exist_ok=True)
    clean=run_trajectory('clean')
    rows=[]
    for flip_round in [0,3]:
        pert=run_trajectory(f'flip_{flip_round}',flip_round=flip_round)
        m=compare(clean,pert,flip_round); rows.append(m)
        print('\nPAIR RESULT')
        for k,v in m.items(): print(f'  {k}: {v}')
    with open('results/summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print('\nDone -> results/summary.csv')

if __name__=='__main__':
    main()
