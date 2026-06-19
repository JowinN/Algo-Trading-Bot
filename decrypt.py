import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# 1. Your inputs
ciphertext_hex = "f34136d9823cb28fa328508ec9649b7eec3ad3cb8b17b281540653a53f660e0f42ae0f9670159895313b471440a0e2ed863c1ed4606a268885579abbb4822bf5"
key_string = "http://www.infosys.com/finacle/index.html"

# Convert hex ciphertext back to raw bytes
ciphertext = bytes.fromhex(ciphertext_hex)

# 2. Derive a proper 256-bit key from your URL string using SHA-256
key = hashlib.sha256(key_string.encode('utf-8')).digest()

# 3. Guessing the Mode of Operation (Assuming AES-256-CBC)
# CBC mode requires an Initialization Vector (IV). 
# In many legacy or basic database integrations, the first 16 bytes of the 
# ciphertext are used as the IV, or it is a static block of 16 null bytes.
iv = ciphertext[:16]  
encrypted_data = ciphertext[16:]

try:
    backend = default_backend()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    decryptor = cipher.decryptor()
    decrypted_bytes = decryptor.update(encrypted_data) + decryptor.finalize()
    
    # Strip PKCS7 padding and decode to readable text
    padding_len = decrypted_bytes[-1]
    plaintext = decrypted_bytes[:-padding_len].decode('utf-8', errors='ignore')
    print("Decrypted plaintext:", plaintext)
except Exception as e:
    print("Decryption failed:", e)

