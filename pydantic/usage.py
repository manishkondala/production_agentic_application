from pydantic import BaseModel, EmailStr, AnyUrl, Field, field_validator, model_validator, computed_field
from typing import List, Dict

class Address(BaseModel):
    city: str
    state: str
    zip: int

class PatientData(BaseModel):
    name: str
    age: int
    weight: float
    height: float
    married: bool
    allergies: list[str]
    email: EmailStr
    contactInfo: dict[str, str]
    address: Address

    @field_validator('email')
    @classmethod
    def valid_email(cls, value):
        valid_domains = ['hdfc.com', 'icici.com']
        domain_name = value.split('@')[-1]

        if domain_name not in valid_domains:
            raise ValueError ("Not a valid domain")
        
        return value

    @field_validator('name')
    @classmethod
    def transformName(cls, value):
        return value.upper()

    @model_validator(mode='after')
    def validate_emergency_contact(cls, model):
        if model.age > 60 and 'emergency' not in model.contactInfo:
            raise ValueError ("Age > 60 and no emergency contact")

        emergency = model.contactInfo.get('emergency')
        if emergency is not None and emergency ==  model.contactInfo.get('phone'):
            raise ValueError ("Emergency contact and phone cannot be the same!!")

        return model

    @computed_field
    @property
    def calculateBMI(self) -> float:
        bmi = round((self.weight) / (self.height ** 2), 2)
        return bmi

def addPatientData(patient_data: PatientData):
    print(patient_data.name)
    print(patient_data.age)
    print(patient_data.weight)
    print(patient_data.height)
    print(patient_data.married)
    print(patient_data.allergies)
    print(patient_data.email)
    print(patient_data.contactInfo)
    print(f"BMI: ", patient_data.calculateBMI)
    print("Address: ", patient_data.address)

address_1 = {"city": "Germantown", "state": "MD", "zip": 20874}
address_2 = {"city": "Redmond", "state": "WA", "zip": 98052}

manish_addr = Address(**address_1)
navya_addr = Address(**address_2)

patient_data_1 = {
    "name": "Manish",
    "age": "26",
    "weight": "90",
    "height": 5.11,
    "married": False,
    "allergies": ["Fish", "Gluten"],
    "contactInfo": {"phone": "+19295267913"},
    "email": "kondalamanish@hdfc.com",
    "address": manish_addr
}

patient_data_2 = {
    "name": "Navya",
    "age": "61",
    "weight": "45",
    "height": 5.1,
    "married": "False",
    "allergies": ["None"],
    "contactInfo": {"phone": "+12272452357", "emergency": "+12272452358"}, 
    "email": "n@icici.com",
    "address": navya_addr
}

patient1 = PatientData(**patient_data_1)
patient2 = PatientData(**patient_data_2)
addPatientData(patient1)
addPatientData(patient2)