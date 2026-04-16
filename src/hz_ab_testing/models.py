"""Pydantic models for the Hazel A/B testing demo."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Segment = Literal["new", "returning", "vip"]
TimingPool = Literal["evening_workday", "weekend_noon", "unknown"]
AgeRange = Literal["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
IncomeBracket = Literal["<50k", "50-75k", "75-100k", "100-150k", "150k+"]
AcquisitionChannel = Literal[
    "organic_search", "paid_social", "referral", "email", "direct",
]
LifestyleInterest = Literal[
    "outdoor_enthusiast", "fashion_forward", "budget_conscious",
    "athleisure", "sustainable_fashion", "minimalist", "streetwear",
]
ProductCategory = Literal[
    "jacket", "sweater", "coat", "boots", "scarf", "gloves",
    "hat", "thermal", "pants", "socks",
]
EngagementType = Literal["open", "click"]


class Product(BaseModel):
    product_id: str
    name: str
    category: ProductCategory
    price: float = Field(gt=0)
    rating: float = Field(ge=0, le=5)
    review_count: int = Field(ge=0)


class Customer(BaseModel):
    customer_id: str
    name: str
    email: str
    segment: Segment
    avg_order_value: float
    total_orders: int
    clv: float  # derived: avg_order_value * total_orders
    last_purchase_days_ago: int
    return_rate: float  # percentage 0-100
    acquisition_channel: AcquisitionChannel
    lifestyle_interests: list[LifestyleInterest]
    preferred_send_time: TimingPool  # ground truth label
    age_range: AgeRange
    income_bracket: IncomeBracket
    city: str
    state: str


class CartItem(BaseModel):
    product_id: str
    product_name: str
    category: ProductCategory
    price: float
    quantity: int = Field(ge=1)


class Cart(BaseModel):
    cart_id: str
    customer_id: str
    items: list[CartItem]
    cart_total: float
    abandoned_at: datetime


class EngagementEvent(BaseModel):
    event_id: str
    customer_id: str
    event_type: EngagementType
    occurred_at: datetime
