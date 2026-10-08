from datetime import datetime
from pydantic import BaseModel


class DemandInput(BaseModel):
    order_timestamp: datetime
    is_weekend: int
    weather: str
    traffic_level: str


class DeliveryInput(BaseModel):
    distance_km: float
    traffic_level: str
    restaurant_preparation_time_minutes: int
    estimated_delivery_time_minutes: int
    delivery_partner_experience_months: int
    delivery_partner_rating: float
    weather: str
    is_weekend: int


class SentimentInput(BaseModel):
    review: str