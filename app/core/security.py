from passlib.context import CryptContext
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict
import jwt
from app.core.config import get_settings

# Bcrypt has a 72-byte password limit. To handle longer passwords,
# we hash them with SHA256 first, then bcrypt the result.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """
    Hash a password for storing.
    
    Bcrypt has a 72-byte limit, so we SHA256 hash first for longer passwords.
    This is a common pattern and doesn't weaken security.
    """
    # Pre-hash with SHA256 if password is long to avoid bcrypt's 72-byte limit
    if len(password.encode('utf-8')) > 72:
        password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a stored password against one provided by user.
    
    Applies the same SHA256 pre-hashing if needed.
    """
    # Pre-hash with SHA256 if password is long (same as in hash_password)
    if len(plain_password.encode('utf-8')) > 72:
        plain_password = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary of claims to encode in the token
        expires_delta: Optional expiration time delta
        
    Returns:
        Encoded JWT token string
    """
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify a JWT access token.
    
    Args:
        token: JWT token string
        
    Returns:
        Dictionary of decoded claims
        
    Raises:
        jwt.PyJWTError: If token is invalid or expired
    """
    settings = get_settings()
    payload = jwt.decode(
        token,
        settings.JWT_SECRET,
        algorithms=[settings.JWT_ALGORITHM]
    )
    return payload