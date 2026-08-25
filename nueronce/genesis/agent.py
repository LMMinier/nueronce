"""Knowledge-gap-driven technical agent controller."""
from dataclasses import dataclass
from typing import List
from .plastic_memory import PlasticMemory

@dataclass
class TaskPlan:
    goal: str
    required_concepts: List[str]
    missing: List[str]
    ready: bool

class TechnicalAgent:
    def __init__(self,memory: PlasticMemory):
        self.memory=memory

    def plan(self,goal,required_concepts):
        missing=[]
        for c in required_concepts:
            r=self.memory.read(c,"procedure")
            if not (r and r.verified and r.confidence>=.9):
                missing.append(c)
        return TaskPlan(goal,list(required_concepts),missing,not missing)

    def record_success(self,skill,procedure,source="agent_experience"):
        return self.memory.write(skill,"procedure",procedure,source=source,
                                 confidence=.97,verified=True,kind="procedural")
