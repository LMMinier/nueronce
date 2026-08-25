from nueronce.genesis import PlasticMemory, CurriculumEngine, Lesson, TechnicalAgent
from nueronce.genesis.domains.coding import algorithm_case
from nueronce.genesis.domains.mathscience import physics

def test_persistent_correction(tmp_path):
    p=tmp_path/"m.json"; m=PlasticMemory(p)
    m.write("Velora","capital_of","Nemi",source="chat")
    r=m.write("Velora","capital_of","Nera",source="correction")
    assert r.version==2
    assert PlasticMemory(p).read("Velora","capital_of").object=="Nera"

def test_verified_skill_unlocks_agent(tmp_path):
    m=PlasticMemory(tmp_path/"m.json"); c=CurriculumEngine(m)
    c.register_verifier("physics",physics)
    lesson=Lesson("physics","mechanics.newton2","F=ma","Compute F=m*a.","verified-seed","physics",("newton2",(3,4)),12)
    assert c.study(lesson)["ok"]
    assert TechnicalAgent(m).plan("solve",["mechanics.newton2"]).ready

def test_missing_skill_blocks_agent(tmp_path):
    m=PlasticMemory(tmp_path/"m.json")
    plan=TechnicalAgent(m).plan("modify repo",["git.branching","python.testing"])
    assert not plan.ready and len(plan.missing)==2

def test_algorithm_transfer():
    assert algorithm_case(("binary_search",([2,4,8,16,32],16)))==3
    assert algorithm_case(("dedupe_order",[3,1,3,2,1]))==[3,1,2]
