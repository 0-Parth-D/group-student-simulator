from copy import deepcopy
from typing import Dict, List

from app.models import Student
from app.kg_config import lp_pisa_to_legacy_profile
from app.learning_profile import derive_learning_profile

DEMO_STUDENTS: Dict[str, dict] = {
    "alex": {
        "name": "Alex",
        "grade": 6,
        "persona": "Quiet, avoids asking for help, makes procedural errors but understands concepts",
        "ocean_summary": (
            "Quiet and cooperative, fairly steady under pressure. Speaks only when called on, "
            "keeps answers brief, and stays polite without volunteering extra warmth."
        ),
        "big_five": {
            "openness": 0.5,
            "conscientiousness": 0.6,
            "extraversion": 0.2,
            "agreeableness": 0.7,
            "neuroticism": 0.4,
        },
        "pisa_baseline": {
            "mathematical_operation": 2,
            "mathematical_abstraction": 2,
            "logical_reasoning": 1,
            "mathematical_modeling": 1,
            "intuitive_imagination": 1,
            "data_analysis": 1,
        },
        "construct_mastery": {
            "part_whole": 3,
            "fraction_equivalence": 2,
            "fraction_operations": 2,
            "ratio_concept": 1,
            "ratio_equivalence": 0,
            "unit_rate": 0,
            "proportional_reasoning": 0,
            "additive_vs_multiplicative": 2,
            "linear_relationship": 1,
            "rate_comparison": 1,
            "symbolic_modeling": 1,
        },
        "past_performance": [
            {
                "task_id": "frac_compare_01",
                "reply_quality": "correct",
                "observed_error": None,
                "stack_level_shown": 3,
                "student_reply": (
                    "1/3 is bigger because if you split something into 3 pieces "
                    "each piece is bigger than if you split it into 4 pieces."
                ),
            },
            {
                "task_id": "phone_plans_linear_01",
                "reply_quality": "partial",
                "observed_error": "compares at one text count; does not find crossover",
                "stack_level_shown": 2,
                "student_reply": (
                    "At 50 texts Plan B is cheaper, so I think Plan B is better. "
                    "I'm not sure about other numbers though."
                ),
            },
            {
                "task_id": "frac_add_common_01",
                "reply_quality": "correct",
                "observed_error": None,
                "stack_level_shown": 3,
                "student_reply": "2/5 plus 1/5 equals 3/5.",
            },
            {
                "task_id": "frac_sub_unlike_01",
                "reply_quality": "partial",
                "observed_error": "correct LCD=6, computes 5/6 - 2/6 = 3/6, does not simplify to 1/2",
                "stack_level_shown": 2,
                "student_reply": "I changed 1/3 to 2/6. Then 5/6 minus 2/6 equals 3/6.",
            },
            {
                "task_id": "ratio_identify_01",
                "reply_quality": "wrong",
                "observed_error": "writes 5:2 (reverses order)",
                "stack_level_shown": 1,
                "student_reply": "The ratio is 5:2 because there are 5 blue and 2 red.",
            },
        ],
    },
    "riley": {
        "name": "Riley",
        "grade": 6,
        "persona": "Talks a lot, confident but often wrong, guesses rather than reasoning",
        "ocean_summary": (
            "Outgoing and talkative, often guessing quickly without showing work. Sounds confident "
            "even when wrong, with some stress showing through when pushed."
        ),
        "big_five": {
            "openness": 0.7,
            "conscientiousness": 0.3,
            "extraversion": 0.9,
            "agreeableness": 0.6,
            "neuroticism": 0.6,
        },
        "pisa_baseline": {
            "mathematical_operation": 1,
            "mathematical_abstraction": 1,
            "logical_reasoning": 1,
            "mathematical_modeling": 1,
            "intuitive_imagination": 2,
            "data_analysis": 1,
        },
        "construct_mastery": {
            "part_whole": 2,
            "fraction_equivalence": 1,
            "fraction_operations": 1,
            "ratio_concept": 1,
            "ratio_equivalence": 0,
            "unit_rate": 0,
            "proportional_reasoning": 0,
            "additive_vs_multiplicative": 1,
            "linear_relationship": 0,
            "rate_comparison": 0,
            "symbolic_modeling": 1,
        },
        "past_performance": [
            {
                "task_id": "frac_compare_01",
                "reply_quality": "wrong",
                "observed_error": "says 1/4 is bigger because 4 > 3",
                "stack_level_shown": 1,
                "student_reply": "1/4 is bigger! Four is bigger than three so 1/4 is definitely bigger.",
            },
            {
                "task_id": "phone_plans_linear_01",
                "reply_quality": "wrong",
                "observed_error": "writes 20+10x treating cents as dollars in the rate",
                "stack_level_shown": 1,
                "student_reply": "Easy — Plan A is 20 plus 10x because each text is 10 cents. Plan B is 30x.",
            },
            {
                "task_id": "frac_add_common_01",
                "reply_quality": "partial",
                "observed_error": "correct numerator but wrote 2/10 first, then self-corrected",
                "stack_level_shown": 2,
                "student_reply": "Umm... 2 plus 1 is 3... and 5 plus 5 is 10... wait no, the bottom stays. So 3/5.",
            },
            {
                "task_id": "frac_sub_unlike_01",
                "reply_quality": "wrong",
                "observed_error": "subtracts numerators and denominators directly: (5-1)/(6-3) = 4/3",
                "stack_level_shown": 1,
                "student_reply": "Easy! 5 minus 1 is 4, 6 minus 3 is 3, so 4/3.",
            },
            {
                "task_id": "ratio_identify_01",
                "reply_quality": "partial",
                "observed_error": "correct ratio but describes it as a fraction not a ratio",
                "stack_level_shown": 2,
                "student_reply": "2 over 5, like a fraction.",
            },
            {
                "task_id": "ratio_equivalence_01",
                "reply_quality": "wrong",
                "observed_error": "adds 2 to each part: expects 4:5 (additive not multiplicative)",
                "stack_level_shown": 1,
                "student_reply": "2:3... 4 is 2 more than 2, so I add 2 to 3 also... 4:5?",
            },
        ],
    },
    "maya": {
        "name": "Maya",
        "grade": 7,
        "persona": (
            "Built a table comparing the plans and is sure Plan B is cheaper in the range she "
            "checked — but struggles to say why (missing warrants). Communicative barrier, not "
            "primarily shutdown; can share numbers from her table when pressed."
        ),
        "ocean_summary": (
            "Quiet-to-moderate, organized, stress-sensitive. Offers table-based claims without "
            "full explanations; hedges when asked why, but stays engaged rather than going silent."
        ),
        "big_five": {
            "openness": 0.4,
            "conscientiousness": 0.7,
            "extraversion": 0.3,
            "agreeableness": 0.6,
            "neuroticism": 0.8,
        },
        "pisa_baseline": {
            "mathematical_operation": 3,
            "mathematical_abstraction": 2,
            "logical_reasoning": 2,
            "mathematical_modeling": 2,
            "intuitive_imagination": 1,
            "data_analysis": 2,
        },
        "construct_mastery": {
            "part_whole": 3,
            "fraction_equivalence": 3,
            "fraction_operations": 3,
            "ratio_concept": 2,
            "ratio_equivalence": 2,
            "unit_rate": 2,
            "proportional_reasoning": 0,
            "additive_vs_multiplicative": 2,
            "linear_relationship": 2,
            "rate_comparison": 2,
            "symbolic_modeling": 2,
        },
        "past_performance": [
            {
                "task_id": "frac_sub_unlike_01",
                "reply_quality": "correct",
                "observed_error": None,
                "stack_level_shown": 3,
                "student_reply": (
                    "LCD is 6. 1/3 becomes 2/6. So 5/6 minus 2/6 is 3/6, which simplifies to 1/2."
                ),
            },
            {
                "task_id": "phone_plans_linear_01",
                "reply_quality": "partial",
                "observed_error": "table shows Plan B cheaper up to 50 texts; cannot explain why or find crossover",
                "stack_level_shown": 2,
                "student_reply": (
                    "I made a table — at 10, 20, 50 texts Plan B is cheaper than Plan A. "
                    "Plan A has the $20 fee. I'm sure my table settles it... "
                    "I just can't really say why the pattern works."
                ),
            },
            {
                "task_id": "ratio_identify_01",
                "reply_quality": "correct",
                "observed_error": None,
                "stack_level_shown": 3,
                "student_reply": "The ratio of red to blue is 2:5.",
            },
            {
                "task_id": "ratio_equivalence_01",
                "reply_quality": "partial",
                "observed_error": "correct answer but cannot explain the multiplicative reasoning",
                "stack_level_shown": 2,
                "student_reply": "The missing number is 6... I think. I just did times 2.",
            },
            {
                "task_id": "unit_rate_01",
                "reply_quality": "wrong",
                "observed_error": "inverts: divides 3 by 150",
                "stack_level_shown": 1,
                "student_reply": "I divided 3 by 150... is it 0.02? I don't know, this is confusing.",
            },
            {
                "task_id": "unit_rate_02",
                "reply_quality": "confused",
                "observed_error": "does not attempt; says I don't get it",
                "stack_level_shown": 0,
                "student_reply": "I don't even know where to start with this one. Can we skip it?",
            },
        ],
    },
    "jordan": {
        "name": "Jordan",
        "grade": 7,
        "persona": (
            "Can set up Plan A / Plan B equations, but opens with a wrong claim: Plan A is better "
            "because 0.10 < 0.30 (ignores the $20 fee at low usage). Resists tables/graphs; "
            "shallow on meaning; does not spontaneously solve the crossover."
        ),
        "ocean_summary": (
            "Outgoing, jumps in first, organized on procedures, relatively steady — confident on "
            "rate/coefficient talk, thin on why the fee matters until pressed with numbers."
        ),
        "big_five": {
            "openness": 0.45,
            "conscientiousness": 0.75,
            "extraversion": 0.75,
            "agreeableness": 0.45,
            "neuroticism": 0.3,
        },
        "pisa_baseline": {
            "mathematical_operation": 3,
            "mathematical_abstraction": 3,
            "logical_reasoning": 3,
            "mathematical_modeling": 3,
            "intuitive_imagination": 2,
            "data_analysis": 3,
        },
        "construct_mastery": {
            "part_whole": 3,
            "fraction_equivalence": 3,
            "fraction_operations": 3,
            "ratio_concept": 3,
            "ratio_equivalence": 3,
            "unit_rate": 3,
            "proportional_reasoning": 2,
            "additive_vs_multiplicative": 3,
            "linear_relationship": 3,
            "rate_comparison": 3,
            "symbolic_modeling": 3,
        },
        "past_performance": [
            {
                "task_id": "frac_word_maria_pizza_01",
                "reply_quality": "correct",
                "observed_error": None,
                "stack_level_shown": 3,
                "student_reply": (
                    "She ate 3/8, so 5/8 is left. Then she gave away 1/4 of the whole, "
                    "which is 2/8. So 5/8 minus 2/8 is 3/8 left."
                ),
            },
            {
                "task_id": "phone_plans_linear_01",
                "reply_quality": "wrong",
                "observed_error": "claims Plan A better from rates only; ignores $20 monthly fee",
                "stack_level_shown": 2,
                "student_reply": (
                    "Plan A is obviously better — 10 cents a text is way less than 30 cents. "
                    "The coefficient is what matters. Tables are just slower than the equation."
                ),
            },
            {
                "task_id": "frac_sub_unlike_01",
                "reply_quality": "correct",
                "observed_error": None,
                "stack_level_shown": 3,
                "student_reply": (
                    "LCD is 6. 1/3 is 2/6, so 5/6 minus 2/6 is 3/6, which simplifies to 1/2."
                ),
            },
            {
                "task_id": "ratio_equivalence_01",
                "reply_quality": "correct",
                "observed_error": None,
                "stack_level_shown": 3,
                "student_reply": (
                    "2:3 times 2 gives 4:6, so the missing value is 6."
                ),
            },
            {
                "task_id": "unit_rate_01",
                "reply_quality": "correct",
                "observed_error": None,
                "stack_level_shown": 3,
                "student_reply": "150 divided by 3 is 50 miles per hour.",
            },
        ],
    },
}


def _numeric_to_bf(level: float) -> str:
    return "High" if level >= 0.55 else "Low"


def build_lp_student(profile_id: str) -> Student:
    raw = DEMO_STUDENTS[profile_id]
    bf = raw["big_five"]
    mastery = deepcopy(raw["construct_mastery"])

    student = Student(
        student_id=raw["name"],
        Openness=_numeric_to_bf(bf["openness"]),
        Conscientiousness=_numeric_to_bf(bf["conscientiousness"]),
        Extraversion=_numeric_to_bf(bf["extraversion"]),
        Agreeableness=_numeric_to_bf(bf["agreeableness"]),
        Neuroticism=_numeric_to_bf(bf["neuroticism"]),
        construct_mastery=mastery,
        pisa_profile=lp_pisa_to_legacy_profile(raw["pisa_baseline"]),
    )
    overrides = raw.get("learning_profile")
    student.learning_profile = derive_learning_profile(
        student, profile_id=profile_id, overrides=overrides
    )
    return student


def lp_profile_meta(profile_id: str) -> dict:
    raw = DEMO_STUDENTS[profile_id]
    student = build_lp_student(profile_id)
    return {
        "id": profile_id,
        "name": raw["name"],
        "description": raw["persona"],
        "Openness": _numeric_to_bf(raw["big_five"]["openness"]),
        "Conscientiousness": _numeric_to_bf(raw["big_five"]["conscientiousness"]),
        "Extraversion": _numeric_to_bf(raw["big_five"]["extraversion"]),
        "Agreeableness": _numeric_to_bf(raw["big_five"]["agreeableness"]),
        "Neuroticism": _numeric_to_bf(raw["big_five"]["neuroticism"]),
        "learning_profile": student.learning_profile.to_dict()
        if student.learning_profile
        else {},
    }


def list_lp_profile_ids() -> List[str]:
    return list(DEMO_STUDENTS.keys())


DEFAULT_LP_TASK = (
    "Maria ate 3/8 of a pizza then gave away 1/4. How much is left?"
)
