from typing import Dict, List

from app.student.demo_students import (
    DEFAULT_LP_TASK,
    build_lp_student,
    list_lp_profile_ids,
    lp_profile_meta,
)
from app.student.models import Student

DEFAULT_TASK = DEFAULT_LP_TASK

PROFILES: List[Dict] = [lp_profile_meta(pid) for pid in list_lp_profile_ids()]


def get_profile(profile_id: str) -> Dict:
    if profile_id not in list_lp_profile_ids():
        raise KeyError(f"Unknown profile: {profile_id}")
    return lp_profile_meta(profile_id)


def make_student(profile_id: str) -> Student:
    return build_lp_student(profile_id)
