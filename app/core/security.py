from passlib.context import CryptContext
import hashlib

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