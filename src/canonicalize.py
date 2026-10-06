"""Canonical InChIKey14 computation matching the CASMI 2026 scorer exactly.

Per the competition's Evaluation page: both the prediction and the answer are
passed through RDKit tautomer canonicalization (pinned 2026.3.3), then
reduced to the first block of their InChIKey (InChIKey14), and compared.
"""
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.inchi import MolToInchiKey
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl

logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")

_enumerator = rdMolStandardize.TautomerEnumerator()


def canonical_inchikey14(smiles: str) -> str | None:
    """Returns the InChIKey14 of the canonical tautomer, or None if unparseable."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        canon_mol = _enumerator.Canonicalize(mol)
    except Exception:
        canon_mol = mol
    try:
        ikey = MolToInchiKey(canon_mol)
    except Exception:
        return None
    if not ikey:
        return None
    return ikey.split("-")[0]
