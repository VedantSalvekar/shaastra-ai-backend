
from dataclasses import dataclass
from typing import List


@dataclass
class LegalSource:
    url: str
    provider: str          
    topic: str            
    subtopic: str          
    description: str = ""  


LEGAL_SOURCES: List[LegalSource] = [
    LegalSource(
        url="https://www.citizensinformation.ie/en/moving-country/visas-for-ireland/student-visas/",
        provider="citizensinformation",
        topic="immigration",
        subtopic="stamps_overview",
        description="Overview of Irish immigration stamps 1, 1A, 1G, 2, 2A, 3, 4, 5",
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


    # Transportation: Leap Cards, Student Leap, Public Transport Refunds
    LegalSource(
        url="https://about.leapcard.ie/about",
        provider="leapcard",
        topic="transport",
        subtopic="leap_card_info",
        description="Leap Card details, usage, and card types.",
    ),

    LegalSource(
        url="https://about.leapcard.ie/fares-and-tickets/student",
        provider="leapcard",
        topic="transport",
        subtopic="student_leap",
        description="Student Leap Card eligibility and discounts.",
    ),

    # Driving and Licensing
    LegalSource(
        url="https://www.ndls.ie/holders-of-foreign-licences.html",
        provider="ndls",
        topic="transport",
        subtopic="foreign_license",
        description="Rules for exchanging or using foreign driving licences in Ireland.",
    ),

    #Health system (HSE + CI)
    LegalSource(
        url="https://www2.hse.ie/services/",
        provider="hse",
        topic="health",
        subtopic="health_services_overview",
        description="HSE services overview including GP, emergency, cards and cover.",
    ),

    LegalSource(
        url="https://www.citizensinformation.ie/en/health/health-overview/",
        provider="citizensinformation",
        topic="health",
        subtopic="healthcare_overview",
        description="Plain-language overview of the Irish healthcare system.",
    ),

    #Housing, Tenancy, rental rights (RTB + CI)
    LegalSource(
        url="https://rtb.ie/renting/rights-responsibilities/tenant-rights-responsibilities/?_gl=1*16udmc1*_up*MQ..*_ga*MTgzOTcxNzY4OC4xNzY1MjE5MDg4*_ga_SQ54Y7X4WL*czE3NjUyMTkwODckbzEkZzEkdDE3NjUyMTkxMDMkajQ0JGwwJGgw",
        provider="rtb",
        topic="housing",
        subtopic="tenant_rights",
        description="Official RTB guide on tenant rights and landlord obligations.",
    ),
    
    LegalSource(
        url="https://www.citizensinformation.ie/en/housing/renting-a-home/",
        provider="citizensinformation",
        topic="housing",
        subtopic="renting_overview",
        description="Plain-language guidance for renting, deposits, choosing property.",
    ),

    LegalSource(
        url="https://www.citizensinformation.ie/en/housing/renting-a-home/tenants-rights-and-responsibilities/",
        provider="citizensinformation",
        topic="housing",
        subtopic="rental_disputes",
        description="Handling disputes, repairs, illegal eviction, rent issues.",
    )


]
