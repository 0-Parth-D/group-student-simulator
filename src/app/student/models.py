from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from app.student.learning_profile import LearningProfile


@dataclass
class Student:
    student_id: str
    Openness: str
    Conscientiousness: str
    Extraversion: str
    Agreeableness: str
    Neuroticism: str
    # LP mastery map (0=missing … 3=good) from knowledge_graph.yaml
    construct_mastery: Dict[str, int] = field(default_factory=dict, repr=False)
    # PISA six mathematical competencies (low / med / high)
    pisa_profile: Dict[str, str] = field(default_factory=dict, repr=False)
    learning_profile: Optional["LearningProfile"] = field(default=None, repr=False)

    def personality_dict(self) -> Dict[str, str]:
        return {
            "Openness": self.Openness,
            "Conscientiousness": self.Conscientiousness,
            "Extraversion": self.Extraversion,
            "Agreeableness": self.Agreeableness,
            "Neuroticism": self.Neuroticism,
        }
