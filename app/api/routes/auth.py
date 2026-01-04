from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

from app.api.deps import get_db, get_current_user
from app.crud.user import create_user, get_user_by_email
from app.schemas.user import UserCreate, UserRead, UserLogin, Token
from app.core.security import create_access_token, verify_password
from app.core.config import get_settings
from app.models.user import User

router = APIRouter()


@router.post("/signup", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def signup(user_in: UserCreate, db: Session = Depends(get_db)) -> UserRead:
    """
    Create a new user account.
    
    Args:
        user_in: User registration data (email, password, full_name)
        db: Database session
        
    Returns:
        Created user data
        
    Raises:
        409: Email already registered
        400: Password too short
    """
    existing = get_user_by_email(db, user_in.email.lower().strip())
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email is already registered.",
        )

    if len(user_in.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters.",
        )

    user = create_user(db, user_in)
    return user


@router.post("/login", response_model=Token)
def login(credentials: UserLogin, db: Session = Depends(get_db)) -> Token:
    """
    Login with email and password to get an access token.
    
    Args:
        credentials: User login credentials (email, password)
        db: Database session
        
    Returns:
        Access token for authenticated requests
        
    Raises:
        401: Invalid credentials or inactive account
    """
    # Find user by email
    user = get_user_by_email(db, credentials.email.lower().strip())
    
    # Verify user exists and password is correct
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    settings = get_settings()
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"user_id": user.id, "email": user.email},
        expires_delta=access_token_expires
    )
    
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserRead)
def get_me(current_user: User = Depends(get_current_user)) -> UserRead:
    """
    Get current authenticated user's profile.
    
    Requires valid JWT token in Authorization header:
    Authorization: Bearer <token>
    
    Args:
        current_user: Current authenticated user (from JWT token)
        
    Returns:
        Current user's profile data
    """
    return current_user
