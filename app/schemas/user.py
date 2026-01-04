from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    email: EmailStr
    full_name: str | None = None


class UserCreate(UserBase):
    password: str  


class UserLogin(BaseModel):
    """Schema for user login request."""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Schema for token response."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema for decoded token data."""
    user_id: int
    email: str


class UserRead(UserBase):
    id: int
    is_active: bool

    class Config:
        orm_mode = True
