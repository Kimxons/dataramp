import ast
import astor
import hashlib
from keyword import iskeyword
from typing import Dict, Set

class CodeSigner(ast.NodeTransformer):
    """Embeds cryptographic signatures in code structure"""
    
    def __init__(self, master_key: str):
        self.master_key = master_key
        self.identifier_map: Dict[str, str] = {}
        self.used_identifiers: Set[str] = set()
        
        # Generate base signature from master key
        self.base_sig = hashlib.blake2b(
            master_key.encode(), 
            digest_size=16
        ).hexdigest()

    def _signed_identifier(self, original: str) -> str:
        """Create deterministic hashed identifier"""
        digest = hashlib.blake2b(
            (original + self.master_key).encode(),
            digest_size=8
        ).hexdigest()
        
        identifier = f"_{self.base_sig[:4]}_{digest}"
        if identifier in self.used_identifiers or iskeyword(identifier):
            identifier += "_"
        self.used_identifiers.add(identifier)
        return identifier

    def _sign_structure(self, node):
        """Recursively apply signature to code structure"""
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Name, ast.Attribute)):
            if hasattr(node, 'name'):
                node.name = self._signed_identifier(node.name)
            if hasattr(node, 'id'):
                node.id = self._signed_identifier(node.id)
            if hasattr(node, 'attr'):
                node.attr = self._signed_identifier(node.attr)
        self.generic_visit(node)
        return node

    visit_FunctionDef = _sign_structure
    visit_ClassDef = _sign_structure
    visit_Name = _sign_structure
    visit_Attribute = _sign_structure

def sign_code(source: str, master_key: str) -> str:
    """Apply base signature to all code identifiers"""
    tree = ast.parse(source)
    CodeSigner(master_key).visit(tree)
    ast.fix_missing_locations(tree)
    return astor.to_source(tree)

def embed_fingerprint(code: str, master_key: str) -> str:
    """Embed invisible cryptographic fingerprint"""
    fingerprint = hashlib.blake2b(
        code.encode() + master_key.encode(),
        digest_size=16
    ).hexdigest()
    
    return (
        f"{code}\n"
        f"# {fingerprint[:8]}... (truncated)\n"
        f"assert __import__('hashlib').blake2b(__import__('sys')._getframe().f_code.co_code"
        f" + {master_key.encode()!r}, digest_size=16).hexdigest() == '{fingerprint}'"
    )

def process_file(input_path: str, master_key: str):
    """Sign and fingerprint a single file"""
    with open(input_path, 'r') as f:
        original = f.read()
    
    signed = sign_code(original, master_key)
    fingerprinted = embed_fingerprint(signed, master_key)
    
    output_path = f"{input_path}.signed"
    with open(output_path, 'w') as f:
        f.write(fingerprinted)
    
    return output_path

if __name__ == "__main__":
    # Use a strong master key from secure storage
    MASTER_KEY = "your-secure-256bit-key-here"  
    
    signed_file = process_file("model_train.py", MASTER_KEY)
    print(f"Signed code generated: {signed_file}")