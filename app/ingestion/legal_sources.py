
from dataclasses import dataclass
from typing import List


@dataclass
class LegalSource:
    url: str
    provider: str          # e.g. "citizensinformation"
    topic: str             # e.g. "immigration"
    subtopic: str          # e.g. "stamp_2"
    description: str = ""  # optional human-readable note


# 🚨 These are just examples – you can refine/expand later.
LEGAL_SOURCES: List[LegalSource] = [
    LegalSource(
        url="https://www.citizensinformation.ie/en/moving-country/visas-for-ireland/student-visas/",
        provider="citizensinformation",
        topic="immigration",
        subtopic="stamps_overview",
        description="Overview of Irish immigration stamps 1, 1A, 1G, 2, 2A, 3, 4, 5",
    ),
    LegalSource(
        url="https://www.citizensinformation.ie/en/moving-country/immigration/students/working-as-a-student/",
        provider="citizensinformation",
        topic="immigration",
        subtopic="student_working_hours",
        description="Rules for working while studying (Stamp 2, Stamp 2A, etc.)",
    ),
    LegalSource(
        url="https://www.citizensinformation.ie/en/moving-country/immigration/permission-to-remain/third-level-graduate-programme/",
        provider="citizensinformation",
        topic="immigration",
        subtopic="stamp_1g_third_level_graduate",
        description="Third Level Graduate Programme (Stamp 1G)",
    ),
    LegalSource(
        url="https://www.citizensinformation.ie/en/employment/employment-rights-and-conditions/employment-rights-and-duties/employee-rights-and-entitlements/",
        provider="citizensinformation",
        topic="employment",
        subtopic="working_hours",
        description="General working hours in Ireland",
    ),
    LegalSource(
        url="https://www.citizensinformation.ie/en/money-and-tax/tax/income-tax/how-your-tax-is-calculated/",
        provider="citizensinformation",
        topic="tax",
        subtopic="income_tax_overview",
        description="Overview of Irish income tax system",
    ),
    LegalSource(
        url="https://www.citizensinformation.ie/en/social-welfare/irish-social-welfare-system/personal-public-service-number/",
        provider="citizensinformation",
        topic="social_welfare",
        subtopic="pps_number",
        description="How to apply for a PPS number",
    ),
]
